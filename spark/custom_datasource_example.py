"""
Custom PySpark DataSource Example: Two String Columns

This example demonstrates how to create a custom PySpark DataSource that generates
two string columns with sample data. This is useful for:
- Creating custom data generators for testing
- Integrating external data sources into Spark
- Building distributed data pipelines with custom logic

Requirements:
    PySpark 4.0.0+ (DataSource API available in PySpark 3.4+)

Usage:
    python spark/custom_datasource_example.py
"""

import sys
from pathlib import Path

# Add utils to path for imports
sys.path.append(str(Path(__file__).parent.parent / "utils"))

from pyspark.sql import SparkSession
from pyspark.sql.datasource import DataSource, DataSourceReader
from spark_utils import create_mlflow_spark_session


class TwoColumnDataSource(DataSource):
    """
    Custom DataSource that generates two string columns.

    Columns:
    - name: A person/entity name
    - description: A description of that entity
    """

    @classmethod
    def name(cls):
        """Return the name used to reference this data source."""
        return "two_column_source"

    def schema(self):
        """
        Define the schema for the data source.

        Returns:
            String representation of the schema
        """
        # Simple string format (required for PySpark 4.0)
        return "name STRING, description STRING"

    def reader(self, schema):
        """
        Create a reader instance for this data source.

        Args:
            schema: The schema to use for reading

        Returns:
            DataSourceReader instance
        """
        return TwoColumnDataSourceReader(schema, self.options)


class TwoColumnDataSourceReader(DataSourceReader):
    """
    Reader implementation for TwoColumnDataSource.

    Generates sample data with names and descriptions.
    """

    def __init__(self, schema, options):
        """
        Initialize the reader.

        Args:
            schema: Schema for the data
            options: Dictionary of options (e.g., num_rows, category)
        """
        self.schema_obj = schema
        self.options = options

        # Extract options with defaults
        self.num_rows = int(options.get("num_rows", 10))
        self.category = options.get("category", "tech")

        # Sample data based on category
        self.data_templates = {
            "tech": [
                ("Python", "High-level programming language known for simplicity"),
                ("Apache Spark", "Unified analytics engine for large-scale data processing"),
                ("MLflow", "Open-source platform for ML lifecycle management"),
                ("Docker", "Platform for developing, shipping, and running applications"),
                ("Kubernetes", "Container orchestration system for automating deployment"),
                ("TensorFlow", "Open-source machine learning framework"),
                ("React", "JavaScript library for building user interfaces"),
                ("PostgreSQL", "Advanced open-source relational database"),
                ("Redis", "In-memory data structure store used as database and cache"),
                ("Kafka", "Distributed event streaming platform")
            ],
            "science": [
                ("Quantum Mechanics", "Fundamental theory describing nature at atomic scales"),
                ("Relativity", "Theory of space, time, and gravitation by Einstein"),
                ("Evolution", "Process of biological change over generations"),
                ("Photosynthesis", "Process by which plants convert light into energy"),
                ("DNA", "Molecule carrying genetic instructions for life"),
                ("Black Holes", "Regions of spacetime with extreme gravitational effects"),
                ("Antibiotics", "Medicines that fight bacterial infections"),
                ("Neurons", "Specialized cells transmitting nerve impulses"),
                ("Climate Change", "Long-term shifts in global temperatures and weather"),
                ("Plate Tectonics", "Theory explaining Earth's lithosphere movement")
            ],
            "business": [
                ("Revenue", "Total income generated from business operations"),
                ("Profit Margin", "Percentage of revenue that becomes profit"),
                ("Market Share", "Portion of market controlled by a company"),
                ("ROI", "Return on Investment - measure of profitability"),
                ("Supply Chain", "Network of entities involved in product creation"),
                ("Customer Acquisition", "Process of gaining new customers"),
                ("Brand Equity", "Commercial value derived from consumer perception"),
                ("Cash Flow", "Movement of money in and out of business"),
                ("Stakeholder", "Person with interest or concern in the business"),
                ("Competitive Advantage", "Attribute allowing superior performance")
            ]
        }

    def read(self, partition):
        """
        Generate data for a specific partition.

        Args:
            partition: Partition information (contains partition.value as index)

        Yields:
            Tuples of (name, description) for each row
        """
        # Get the appropriate data template
        templates = self.data_templates.get(self.category, self.data_templates["tech"])

        # Generate rows for this partition
        partition_id = partition.value if hasattr(partition, 'value') else 0

        for i in range(self.num_rows):
            # Cycle through templates if needed
            template_idx = (partition_id * self.num_rows + i) % len(templates)
            name, description = templates[template_idx]

            # Add partition and row info for uniqueness
            unique_name = f"{name} (P{partition_id}R{i})"
            enhanced_desc = f"[Partition {partition_id}, Row {i}] {description}"

            yield (unique_name, enhanced_desc)


def demo_basic_usage():
    """Demonstrate basic usage of the custom DataSource."""

    print("=" * 70)
    print("📊 Custom PySpark DataSource: Two String Columns Demo")
    print("=" * 70)

    # Create Spark session
    print("\n🚀 Creating Spark session...")
    spark = create_mlflow_spark_session(app_name="CustomDataSource-Demo")

    try:
        # Register the custom data source
        print("📝 Registering custom DataSource...")
        spark.dataSource.register(TwoColumnDataSource)
        print("✅ DataSource registered as 'two_column_source'")

        # Example 1: Basic usage with default options
        print("\n" + "=" * 70)
        print("Example 1: Basic Usage (Tech Category)")
        print("=" * 70)

        df1 = spark.read.format("two_column_source").load()

        print(f"\n📊 Schema:")
        df1.printSchema()

        print(f"\n📋 Sample Data (showing 10 rows):")
        df1.show(10, truncate=False)

        print(f"\n📈 Total rows: {df1.count()}")

        # Example 2: Custom number of rows
        print("\n" + "=" * 70)
        print("Example 2: Custom Row Count (5 rows)")
        print("=" * 70)

        df2 = spark.read.format("two_column_source") \
            .option("num_rows", 5) \
            .load()

        print(f"\n📋 Data:")
        df2.show(truncate=False)

        # Example 3: Different category (Science)
        print("\n" + "=" * 70)
        print("Example 3: Science Category")
        print("=" * 70)

        df3 = spark.read.format("two_column_source") \
            .option("num_rows", 5) \
            .option("category", "science") \
            .load()

        print(f"\n📋 Data:")
        df3.show(truncate=False)

        # Example 4: Business category
        print("\n" + "=" * 70)
        print("Example 4: Business Category")
        print("=" * 70)

        df4 = spark.read.format("two_column_source") \
            .option("num_rows", 5) \
            .option("category", "business") \
            .load()

        print(f"\n📋 Data:")
        df4.show(truncate=False)

        # Example 5: DataFrame operations
        print("\n" + "=" * 70)
        print("Example 5: DataFrame Operations")
        print("=" * 70)

        df5 = spark.read.format("two_column_source") \
            .option("num_rows", 3) \
            .option("category", "tech") \
            .load()

        # Filter operation
        print("\n🔍 Filter: Names containing 'Python'")
        df5.filter(df5.name.contains("Python")).show(truncate=False)

        # Select operation
        print("\n📝 Select: Just names")
        df5.select("name").show(truncate=False)

        # Add column operation
        print("\n➕ Add column: name_length")
        from pyspark.sql.functions import length
        df5.withColumn("name_length", length("name")).show(truncate=False)

        print("\n" + "=" * 70)
        print("✅ Demo completed successfully!")
        print("=" * 70)

    finally:
        print("\n🛑 Stopping Spark session...")
        spark.stop()
        print("✅ Spark session stopped")


def demo_with_mlflow():
    """Demonstrate integration with MLflow tracking."""

    print("=" * 70)
    print("📊 Custom DataSource + MLflow Integration Demo")
    print("=" * 70)

    import mlflow
    from loader import load_mlflow_setup

    # Setup MLflow
    mlflow_setup = load_mlflow_setup()
    mlflow_setup.setup_mlflow_tracking(
        tracking_uri="file:./mlruns",
        experiment_name="custom-datasource-demo",
        enable_autolog=False
    )

    # Create Spark session
    spark = create_mlflow_spark_session(app_name="CustomDataSource-MLflow")

    try:
        # Register data source
        spark.dataSource.register(TwoColumnDataSource)

        with mlflow.start_run(run_name="custom_datasource_test"):
            # Log parameters
            mlflow.log_param("data_source", "two_column_source")
            mlflow.log_param("category", "tech")
            mlflow.log_param("num_rows", 10)

            # Read data
            df = spark.read.format("two_column_source") \
                .option("num_rows", 10) \
                .option("category", "tech") \
                .load()

            # Calculate metrics
            row_count = df.count()
            distinct_names = df.select("name").distinct().count()

            # Log metrics
            mlflow.log_metric("total_rows", row_count)
            mlflow.log_metric("distinct_names", distinct_names)

            print(f"\n📊 Results:")
            print(f"  Total rows: {row_count}")
            print(f"  Distinct names: {distinct_names}")

            # Show sample
            print(f"\n📋 Sample data:")
            df.show(5, truncate=False)

            print(f"\n✅ MLflow Run completed: {mlflow.active_run().info.run_id}")

    finally:
        spark.stop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Custom PySpark DataSource Demo with Two String Columns'
    )
    parser.add_argument(
        '--mode',
        choices=['basic', 'mlflow'],
        default='basic',
        help='Demo mode: basic or mlflow integration (default: basic)'
    )

    args = parser.parse_args()

    if args.mode == 'basic':
        demo_basic_usage()
    else:
        demo_with_mlflow()
