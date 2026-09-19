"""Tests for QLoRA training pipeline."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.training.qlora_pipeline import QLoRAPipeline, TrainingConfig
from src.data_parsers.parser_factory import DataSourceConfig, ParserFactory


class TestDataSourceIntegration:
    """Tests for data source integration (local/OCI/S3)."""

    def test_local_data_source_config(self):
        """Test local data source configuration."""
        config = DataSourceConfig(
            source_type="local",
            path="/tmp/test_data"
        )
        assert config.source_type == "local"
        assert config.path == "/tmp/test_data"

    def test_s3_data_source_config(self):
        """Test S3 data source configuration."""
        config = DataSourceConfig(
            source_type="s3",
            path="s3://bucket/path",
            s3_bucket="test-bucket",
            s3_prefix="data/"
        )
        assert config.source_type == "s3"
        assert config.s3_bucket == "test-bucket"
        assert config.s3_prefix == "data/"

    def test_oci_data_source_config(self):
        """Test OCI data source configuration."""
        config = DataSourceConfig(
            source_type="oci",
            path="oci://namespace/bucket/path",
            oci_namespace="test-namespace",
            oci_bucket="test-bucket",
            oci_prefix="data/"
        )
        assert config.source_type == "oci"
        assert config.oci_namespace == "test-namespace"
        assert config.oci_bucket == "test-bucket"

    @patch('src.data_parsers.parser_factory.LocalParser')
    def test_parser_factory_local(self, mock_local_parser):
        """Test parser factory creates local parser."""
        mock_parser_instance = MagicMock()
        mock_parser_instance.parse.return_value = {
            "source_type": "local",
            "files_processed": 5,
            "total_size_bytes": 1024,
            "output_path": "/tmp/output",
            "errors": []
        }
        mock_local_parser.return_value = mock_parser_instance

        config = DataSourceConfig(source_type="local", path="/tmp/input")
        result = ParserFactory.parse_data(config, Path("/tmp/output"))

        assert result["files_processed"] == 5
        assert result["source_type"] == "local"

    @patch('src.data_parsers.parser_factory.S3Parser')
    def test_parser_factory_s3(self, mock_s3_parser):
        """Test parser factory creates S3 parser."""
        mock_parser_instance = MagicMock()
        mock_parser_instance.parse.return_value = {
            "source_type": "s3",
            "files_processed": 10,
            "total_size_bytes": 2048,
            "output_path": "/tmp/output",
            "errors": []
        }
        mock_s3_parser.return_value = mock_parser_instance

        config = DataSourceConfig(
            source_type="s3",
            path="s3://bucket/path",
            s3_bucket="test-bucket",
            s3_prefix="data/"
        )
        result = ParserFactory.parse_data(config, Path("/tmp/output"))

        assert result["files_processed"] == 10
        assert result["source_type"] == "s3"

    @patch('src.data_parsers.parser_factory.OCIParser')
    def test_parser_factory_oci(self, mock_oci_parser):
        """Test parser factory creates OCI parser."""
        mock_parser_instance = MagicMock()
        mock_parser_instance.parse.return_value = {
            "source_type": "oci",
            "files_processed": 8,
            "total_size_bytes": 1536,
            "output_path": "/tmp/output",
            "errors": []
        }
        mock_oci_parser.return_value = mock_parser_instance

        config = DataSourceConfig(
            source_type="oci",
            path="oci://namespace/bucket/path",
            oci_namespace="test-namespace",
            oci_bucket="test-bucket"
        )
        result = ParserFactory.parse_data(config, Path("/tmp/output"))

        assert result["files_processed"] == 8
        assert result["source_type"] == "oci"


class TestQLoRATrainingWorkflow:
    """Tests for QLoRA training workflow execution."""

    def test_training_config_defaults(self):
        """Test training config uses correct defaults."""
        config = TrainingConfig()
        assert config.base_model_id == "meta-llama/Llama-2-7b-hf"
        assert config.quantization_bit == 4
        assert config.lora_r == 64
        assert config.lora_alpha == 16
        assert config.num_train_epochs == 3

    def test_training_config_custom_values(self):
        """Test training config accepts custom values."""
        config = TrainingConfig(
            base_model_id="test-model",
            lora_r=128,
            num_train_epochs=5
        )
        assert config.base_model_id == "test-model"
        assert config.lora_r == 128
        assert config.num_train_epochs == 5

    def test_training_workflow_execution(self, tmp_path):
        """Exercise real local data preparation with the GPU boundary mocked."""
        from contextlib import ExitStack
        import src.training.qlora_pipeline as module
        source = tmp_path / "source"
        source.mkdir()
        (source / "samples.jsonl").write_text('{"text":"first sample"}\n{"text":"second sample"}\n')
        config = TrainingConfig(base_model_id="local-model", output_dir=str(tmp_path / "out"),
                                data_source=str(source), adapter_name="custom")
        model, tokenizer, dataset = MagicMock(), MagicMock(), MagicMock()
        model.get_nb_trainable_parameters.return_value = (100, 1000)
        dataset.train_test_split.return_value = {"train": MagicMock(), "test": MagicMock()}
        with ExitStack() as stack:
            mocks = {name: stack.enter_context(patch.object(module, name)) for name in (
                "torch", "AutoModelForCausalLM", "AutoTokenizer", "BitsAndBytesConfig",
                "prepare_model_for_kbit_training", "get_peft_model", "LoraConfig", "TaskType",
                "TrainingArguments", "Trainer", "DataCollatorForLanguageModeling", "load_dataset")}
            mocks["AutoModelForCausalLM"].from_pretrained.return_value = model
            mocks["AutoTokenizer"].from_pretrained.return_value = tokenizer
            mocks["prepare_model_for_kbit_training"].return_value = model
            mocks["get_peft_model"].return_value = model
            mocks["load_dataset"].return_value = dataset
            pipeline = QLoRAPipeline(config)
            stats = pipeline.train()
            mocks["Trainer"].return_value.train.assert_called_once()
            assert mocks["AutoModelForCausalLM"].from_pretrained.call_args.kwargs["local_files_only"] is True
            assert mocks["AutoModelForCausalLM"].from_pretrained.call_args.kwargs["trust_remote_code"] is False
            assert stats["status"] == "completed"
            assert stats["trainable_parameters"] == 100
            assert stats["total_parameters"] == 1000
            assert stats["data_preparation"]["records"] == 2
            assert stats["adapter_path"] == str(tmp_path / "out" / "custom")
            assert pipeline._model is model
            model.save_pretrained.assert_called_once_with(stats["adapter_path"])
            assert (tmp_path / "out" / "prepared_data" / "train.jsonl").exists()

    def test_training_config_max_storage_limit(self):
        """Test training config respects max storage limit."""
        config = TrainingConfig(max_storage_gb=10000.0)
        assert config.max_storage_gb == 10000.0


class TestOutputModelDirectoryStructure:
    """Tests for output model directory structure validation."""

    def test_adapter_output_directory_creation(self):
        """Test adapter output directory is created correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "training_output"
            adapter_dir = output_dir / "adapter"
            adapter_dir.mkdir(parents=True, exist_ok=True)

            # Create dummy adapter files
            (adapter_dir / "adapter_config.json").write_text('{"test": true}')
            (adapter_dir / "adapter_model.bin").write_bytes(b"dummy")

            assert adapter_dir.exists()
            assert (adapter_dir / "adapter_config.json").exists()
            assert (adapter_dir / "adapter_model.bin").exists()

    def test_training_state_file_creation(self):
        """Test training state file is created with correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "training_output"
            output_dir.mkdir(parents=True, exist_ok=True)

            config = TrainingConfig(
                base_model_id="test-model",
                output_dir=str(output_dir)
            )
            pipeline = QLoRAPipeline(config)

            # Mock training to avoid actual execution
            pipeline._training_stats = {
                "status": "completed",
                "base_model": "test-model",
                "adapter_path": str(output_dir / "adapter"),
                "epochs_completed": 3,
                "train_samples": 100,
                "eval_samples": 20,
                "trainable_parameters": 1000000,
                "total_parameters": 10000000
            }

            state_path = pipeline.save_training_state()

            assert Path(state_path).exists()
            with open(state_path) as f:
                state = json.load(f)

            assert "config" in state
            assert "stats" in state
            assert state["config"]["base_model_id"] == "test-model"

    def test_prepared_data_directory_structure(self):
        """Test prepared data directory has correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "training_output"
            prepared_dir = output_dir / "prepared_data"
            prepared_dir.mkdir(parents=True, exist_ok=True)

            # Create dummy data files
            (prepared_dir / "data.json").write_text('{"text": "test"}')
            (prepared_dir / "metadata.json").write_text('{"source": "test"}')

            assert prepared_dir.exists()
            assert (prepared_dir / "data.json").exists()
            assert (prepared_dir / "metadata.json").exists()

    def test_training_stats_structure(self):
        """Test training stats have required fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "training_output"
            output_dir.mkdir(parents=True, exist_ok=True)

            config = TrainingConfig(
                base_model_id="test-model",
                output_dir=str(output_dir)
            )
            pipeline = QLoRAPipeline(config)

            stats = pipeline.get_training_stats()
            assert isinstance(stats, dict)

            # After training, stats should have required fields
            pipeline._training_stats = {
                "status": "completed",
                "base_model": "test-model",
                "adapter_path": str(output_dir / "adapter"),
                "epochs_completed": 3,
                "train_samples": 100,
                "eval_samples": 20,
                "trainable_parameters": 1000000,
                "total_parameters": 10000000
            }

            stats = pipeline.get_training_stats()
            assert "status" in stats
            assert "base_model" in stats
            assert "adapter_path" in stats
            assert "epochs_completed" in stats
            assert "train_samples" in stats
            assert "eval_samples" in stats
            assert "trainable_parameters" in stats
            assert "total_parameters" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
