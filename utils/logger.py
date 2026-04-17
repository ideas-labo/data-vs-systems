import csv
import os
import pandas as pd
from typing import List, Tuple
import json


class Logger:
    def __init__(self, base_dir: str, algorithm: str, system: str, scale: str, header: List[str]):
        self.header = header  # n-1 config names + 1 perf name

        # construct log directory
        self.log_dir = os.path.join(base_dir, algorithm, system, scale)
        os.makedirs(self.log_dir, exist_ok=True)

    def log_generation(self, generation: int, population: List[Tuple[List, float]]):
        """
        save current generation's population to gen{n}.csv
        """
        filename = os.path.join(self.log_dir, f"gen{generation}.csv")
        self._write_csv(filename, population)

    def log_all_results(self, all_results: List[Tuple[List, float]]):
        """
        save all configs and corresponding performance during search to all_results.csv
        """
        filename = os.path.join(self.log_dir, "all_results.csv")
        self._write_csv(filename, all_results)

    def _write_csv(self, filepath: str, data: List[Tuple[List, float]]):
        """
        general writer: config + performance → dataframe + header
        """
        rows = [cfg + [perf] for cfg, perf in data]
        df = pd.DataFrame(rows, columns=self.header)
        df.to_csv(filepath, index=False)

    def store_runtime_to_csv(self, runtime):
        """
        Record the total runtime of the algorithm.
        :param runtime: Total runtime in seconds
        :param log_path: Path to save the CSV file
        """

        file_path = os.path.join(self.log_dir, 'runtime_record.csv')

        # Check if the file exists
        file_exists = os.path.isfile(file_path)

        with open(file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                # Write the header only if the file doesn't exist
                writer.writerow(['runtime_seconds'])
            writer.writerow([runtime])

