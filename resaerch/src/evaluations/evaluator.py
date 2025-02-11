"""
Main script for evaluating models on various benchmarks.
Takes a YAML configuration file that specifies:
- model details (name/path)
- which benchmarks to run
- output preferences
- other evaluation parameters
"""

import argparse
import yaml
import os
import json
from datetime import datetime
from typing import Dict, Any

import benchmarks

class Evaluator:
    """
    Main class for handling model evaluations across different benchmarks
    """
    def __init__(self, config_path: str):
        """
        Initialize evaluator with configuration from YAML file

        Expected YAML structure:
        ```yaml
        model:
          name: "path/to/model/or/huggingface/id"
          batch_size: 32

        benchmarks:
          - name: "glue"
            enabled: true
          - name: "ruglue"
            enabled: false

        output:
          save_path: "path/to/save/results"
          format: ["json", "csv"]  # supported formats
          detailed: true  # whether to include detailed metrics
        ```
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self._validate_config()
        self.results = {}

    def _validate_config(self):
        """Validate that the config has all required fields"""
        required_fields = {
            'model': ['name', 'batch_size'],
            'benchmarks': None,  # List of dicts with 'name' and 'enabled'
            'output': ['save_path', 'format']
        }

        for field, subfields in required_fields.items():
            if field not in self.config:
                raise ValueError(f"Missing required field '{field}' in config")

            if subfields:
                for subfield in subfields:
                    if subfield not in self.config[field]:
                        raise ValueError(f"Missing required subfield '{subfield}' in '{field}'")

        if not self.config['benchmarks']:
            raise ValueError("At least one benchmark must be specified")

    def run_evaluations(self):
        """Run all enabled benchmarks"""
        model_name = self.config['model']['name']
        batch_size = self.config['model']['batch_size']

        for benchmark in self.config['benchmarks']:
            if benchmark['enabled']:
                print(f"\nEvaluating on {benchmark['name'].upper()}...")

                if benchmark['name'].lower() == 'glue':
                    results = benchmarks.evaluate_on_glue(
                        model_file=model_name,
                        batch_size=batch_size
                    )
                    self.results['glue'] = results

                elif benchmark['name'].lower() == 'ruglue':
                    # TODO: Once implemented
                    # results = benchmarks.evaluate_on_ruglue(
                    #     model_file=model_name,
                    #     batch_size=batch_size
                    # )
                    # self.results['ruglue'] = results
                    print("RUGLUE evaluation not yet implemented")

                else:
                    print(f"Unknown benchmark: {benchmark['name']}")

    def save_results(self):
        """Save results in specified formats"""
        if not self.results:
            print("No results to save!")
            return

        # Create timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Ensure output directory exists
        os.makedirs(self.config['output']['save_path'], exist_ok=True)

        # Base filename without extension
        base_filename = os.path.join(
            self.config['output']['save_path'],
            f"eval_results_{timestamp}"
        )

        # Save in each specified format
        for format_type in self.config['output']['format']:
            if format_type.lower() == 'json':
                output_file = f"{base_filename}.json"
                with open(output_file, 'w') as f:
                    json.dump(self.results, f, indent=2)
                print(f"Results saved to {output_file}")

            elif format_type.lower() == 'csv':
                # TODO: Implement CSV output format
                print("CSV output format not yet implemented")

            else:
                print(f"Unsupported output format: {format_type}")

    def print_summary(self):
        """Print a summary of the results to console"""
        if not self.results:
            print("No results to display!")
            return

        print("\n=== Evaluation Results Summary ===")

        for benchmark, results in self.results.items():
            print(f"\n{benchmark.upper()} Results:")

            if isinstance(results, dict):
                # If detailed output is requested
                if self.config['output'].get('detailed', False):
                    for metric, value in results.items():
                        print(f"  {metric}: {value:.4f}")
                else:
                    # Just print the overall score if available
                    if f"{benchmark}_overall_score" in results:
                        print(f"  Overall Score: {results[f'{benchmark}_overall_score']:.4f}")
                    else:
                        print("  No overall score available")
            else:
                print(f"  Score: {results:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate models on various benchmarks")
    parser.add_argument('config', help='Path to YAML configuration file')
    args = parser.parse_args()

    # Initialize and run evaluator
    evaluator = Evaluator(args.config)
    evaluator.run_evaluations()
    evaluator.save_results()
    evaluator.print_summary()

if __name__ == "__main__":
    main()
