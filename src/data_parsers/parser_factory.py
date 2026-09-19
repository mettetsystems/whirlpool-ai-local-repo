"""Parser factory for preprocessing training data from various sources."""

import os
import json
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass



@dataclass
class DataSourceConfig:
    """Configuration for a data source."""
    source_type: str  # 'local', 's3', 'oci'
    path: str
    s3_bucket: Optional[str] = None
    s3_prefix: Optional[str] = None
    oci_namespace: Optional[str] = None
    oci_bucket: Optional[str] = None
    oci_prefix: Optional[str] = None
    oci_auth_config: Optional[str] = None


class DataParser(ABC):
    """Abstract base class for data parsers."""

    @abstractmethod
    def parse(self, source_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Parse data from source and write to output directory.

        Args:
            source_path: Path to the source data.
            output_dir: Directory to write parsed data.

        Returns:
            Dictionary containing parsing statistics and metadata.
        """
        pass


class LocalParser(DataParser):
    """Parser for local directory data sources."""

    def parse(self, source_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Parse local directory data.

        Args:
            source_path: Path to the local source directory.
            output_dir: Directory to write parsed data.

        Returns:
            Dictionary containing parsing statistics.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        stats = {
            "source_type": "local",
            "source_path": str(source_path),
            "files_processed": 0,
            "total_size_bytes": 0,
            "output_path": str(output_dir),
            "errors": []
        }

        if not source_path.exists():
            stats["errors"].append(f"Source path does not exist: {source_path}")
            return stats

        if output_dir.resolve().is_relative_to(source_path.resolve()):
            raise ValueError("Output directory must be outside the source directory")

        for item in source_path.rglob("*"):
            if item.is_file():
                try:
                    # Copy file to output directory preserving structure
                    dest_path = output_dir / item.relative_to(source_path)
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, dest_path)
                    stats["files_processed"] += 1
                    stats["total_size_bytes"] += item.stat().st_size
                except Exception as e:
                    stats["errors"].append(f"Error processing {item}: {str(e)}")

        return stats


class S3Parser(DataParser):
    """Parser for AWS S3 data sources."""

    def __init__(self, aws_access_key_id: Optional[str] = None,
                 aws_secret_access_key: Optional[str] = None,
                 region_name: str = "us-east-1"):
        """Initialize S3 parser.

        Args:
            aws_access_key_id: AWS access key ID.
            aws_secret_access_key: AWS secret access key.
            region_name: AWS region name.
        """
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.region_name = region_name
        self._client = None

    @property
    def client(self):
        """Lazy initialization of S3 client."""
        if self._client is None:
            import boto3
            self._client = boto3.client(
                "s3",
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.region_name
            )
        return self._client

    def parse(self, source_config: DataSourceConfig, output_dir: Path) -> Dict[str, Any]:
        """Parse S3 bucket data.

        Args:
            source_config: S3 source configuration.
            output_dir: Directory to write parsed data.

        Returns:
            Dictionary containing parsing statistics.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        stats = {
            "source_type": "s3",
            "bucket": source_config.s3_bucket,
            "prefix": source_config.s3_prefix or "",
            "files_processed": 0,
            "total_size_bytes": 0,
            "output_path": str(output_dir),
            "errors": []
        }

        try:
            if not source_config.s3_bucket:
                raise ValueError("Specify an S3 bucket")
            prefix = source_config.s3_prefix or ""
            paginator = self.client.get_paginator("list_objects_v2")
            pages = paginator.paginate(
                Bucket=source_config.s3_bucket,
                Prefix=prefix
            )

            for page in pages:
                if "Contents" not in page:
                    continue

                for obj in page["Contents"]:
                    key = obj["Key"]
                    try:
                        # Download object
                        dest_path = safe_destination(output_dir, key)
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        self.client.download_file(
                            source_config.s3_bucket,
                            key,
                            str(dest_path)
                        )
                        stats["files_processed"] += 1
                        stats["total_size_bytes"] += obj["Size"]
                    except Exception as e:
                        stats["errors"].append(f"Error downloading {key}: {str(e)}")

        except Exception as e:
            stats["errors"].append(f"S3 client error: {str(e)}")

        return stats


class OCIParser(DataParser):
    """Parser for Oracle Cloud Infrastructure (OCI) Object Storage."""

    def __init__(self, oci_config_path: Optional[str] = None):
        """Initialize OCI parser.

        Args:
            oci_config_path: Path to OCI config file.
        """
        self.oci_config_path = oci_config_path or os.path.expanduser("~/.oci/config")
        self._client = None

    @property
    def client(self):
        """Lazy initialization of OCI client."""
        if self._client is None:
            try:
                import oci
                self._client = oci.config.from_file(
                    config_location=self.oci_config_path
                )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize OCI client: {str(e)}")
        return self._client

    def parse(self, source_config: DataSourceConfig, output_dir: Path) -> Dict[str, Any]:
        """Parse OCI object storage data.

        Args:
            source_config: OCI source configuration.
            output_dir: Directory to write parsed data.

        Returns:
            Dictionary containing parsing statistics.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        stats = {
            "source_type": "oci",
            "namespace": source_config.oci_namespace,
            "bucket": source_config.oci_bucket,
            "prefix": source_config.oci_prefix or "",
            "files_processed": 0,
            "total_size_bytes": 0,
            "output_path": str(output_dir),
            "errors": []
        }

        try:
            if not source_config.oci_bucket:
                raise ValueError("Specify an OCI bucket")
            import oci
            from oci.object_storage import ObjectStorageClient

            # Get object storage client
            object_storage = ObjectStorageClient(self.client)
            namespace = source_config.oci_namespace or object_storage.get_namespace().data

            # List objects in bucket
            list_objects_response = oci.pagination.list_call_get_all_results(
                object_storage.list_objects,
                namespace=namespace,
                bucket_name=source_config.oci_bucket,
                prefix=source_config.oci_prefix or ""
            )

            for obj in list_objects_response.data.objects:
                try:
                    # Download object
                    dest_path = safe_destination(output_dir, obj.name)
                    dest_path.parent.mkdir(parents=True, exist_ok=True)

                    download_response = object_storage.get_object(
                        namespace=namespace,
                        bucket_name=source_config.oci_bucket,
                        object_name=obj.name
                    )

                    with open(str(dest_path), "wb") as f:
                        f.write(download_response.data.read())

                    stats["files_processed"] += 1
                    stats["total_size_bytes"] += obj.size
                except Exception as e:
                    stats["errors"].append(f"Error downloading {obj.name}: {str(e)}")

        except Exception as e:
            stats["errors"].append(f"OCI client error: {str(e)}")

        return stats


class ParserFactory:
    """Factory for creating appropriate data parsers."""

    PARSERS = {
        "local": LocalParser,
        "s3": S3Parser,
        "oci": OCIParser
    }

    @classmethod
    def create_parser(cls, source_type: str, **kwargs) -> DataParser:
        """Create a parser for the specified source type.

        Args:
            source_type: Type of data source ('local', 's3', 'oci').
            **kwargs: Additional arguments for parser initialization.

        Returns:
            Configured DataParser instance.

        Raises:
            ValueError: If source_type is not supported.
        """
        if source_type not in cls.PARSERS:
            raise ValueError(
                f"Unsupported source type: {source_type}. "
                f"Supported types: {list(cls.PARSERS.keys())}"
            )

        parser_class = {"local": LocalParser, "s3": S3Parser, "oci": OCIParser}[source_type]
        allowed = {"local": (), "s3": ("aws_access_key_id", "aws_secret_access_key", "region_name"),
                   "oci": ("oci_config_path",)}[source_type]
        return parser_class(**{key: value for key, value in kwargs.items() if key in allowed})

    @classmethod
    def parse_data(
        cls,
        source_config: DataSourceConfig,
        output_dir: Union[str, Path]
    ) -> Dict[str, Any]:
        """Parse data from a configured source to output directory.

        Args:
            source_config: DataSourceConfig with source details.
            output_dir: Directory to write parsed data.

        Returns:
            Dictionary containing parsing statistics and metadata.
        """
        output_path = Path(output_dir)

        parser = cls.create_parser(
            source_type=source_config.source_type,
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
            oci_config_path=source_config.oci_auth_config
        )

        if source_config.source_type == "local":
            return parser.parse(Path(source_config.path), output_path)
        else:
            return parser.parse(source_config, output_path)

    @classmethod
    def validate_source(cls, source_config: DataSourceConfig) -> bool:
        """Validate that a data source is accessible.

        Args:
            source_config: DataSourceConfig to validate.

        Returns:
            True if source is accessible, False otherwise.
        """
        try:
            if source_config.source_type == "local":
                return Path(source_config.path).exists()

            elif source_config.source_type == "s3":
                parser = cls.create_parser(
                    source_type="s3",
                    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
                    region_name=os.environ.get("AWS_REGION", "us-east-1")
                )
                # Try to list objects to validate connection
                parser.client.head_bucket(Bucket=source_config.s3_bucket)
                return True

            elif source_config.source_type == "oci":
                parser = cls.create_parser(
                    source_type="oci",
                    oci_config_path=source_config.oci_auth_config
                )
                # Validation happens during client initialization
                return True

        except Exception:
            return False

        return False


def safe_destination(output_dir: Path, key: str) -> Path:
    destination = (output_dir / key).resolve()
    if not destination.is_relative_to(output_dir.resolve()):
        raise ValueError(f"Object path escapes output directory: {key}")
    return destination


def prepare_text_dataset(source_dir: Path, output_file: Path) -> int:
    """Normalize text/JSON data; use optional Docling for documents and images."""
    records = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in (".txt", ".md"):
            values = [{"text": path.read_text(encoding="utf-8")}]
        elif suffix in (".json", ".jsonl"):
            content = path.read_text(encoding="utf-8")
            values = [json.loads(line) for line in content.splitlines() if line.strip()] if suffix == ".jsonl" else json.loads(content)
            if isinstance(values, dict):
                values = [values]
        elif suffix in (".pdf", ".docx", ".pptx", ".html", ".png", ".jpg", ".jpeg", ".tiff"):
            try:
                from docling.document_converter import DocumentConverter
            except ImportError as exc:
                raise RuntimeError("Install requirements-documents.txt to parse documents or images") from exc
            values = [{"text": DocumentConverter().convert(path).document.export_to_markdown()}]
        else:
            raise ValueError(f"Unsupported training file: {path.name}. Use text, JSON, or supported documents/images.")
        if not isinstance(values, list):
            raise ValueError(f"Expected a JSON record or list in {path}")
        for value in values:
            if not isinstance(value, dict) or not isinstance(value.get("text"), str):
                raise ValueError(f"Training records must contain a text string: {path}")
            if value["text"].strip():
                records.append({"text": value["text"]})
    if len(records) < 2:
        raise ValueError("Training requires at least two nonempty text records")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)
