import yaml

from datetime import datetime

from productive_agents.agents.appworld.agent import AppWorldAgent, AppWorldAgentConfig
from productive_agents.agents.officebench.agent import OfficeBenchAgent, OfficeBenchAgentConfig
from productive_agents.agents.smolagents.agent import SmolagentsAgent

from pathlib import Path

from typing import List, Dict


WORKING_DIR = Path.cwd()
EXPERIMENT_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


class BenchmarkRunner: 
    
    def __init__(
        self, 
        global_config_path: Path, 
        benchmark_config_path: Path, 
        experiment_config_path: Path, 
        output_path: Path = None
    ): 
        
        with open(global_config_path, "r") as f: 
            self.global_config = yaml.save_load(global_config_path)
            
        with open(benchmark_config_path, "r") as f: 
            self.benchmark_config = yaml.save_load(benchmark_config_path)
            
        with open(experiment_config_path, "r") as f: 
            self.experiment_config = yaml.save_load(experiment_config_path)
            
        self.config = {}
        self.config.update(self.global_config)
        self.config.update(self.benchmark_config)
        self.config.update(self.experiment_config)
        
        self.config["output_path"] = Path(output_path) if output_path is not None else Path(f"{WORKING_DIR}/outputs/{self.benchmark_config['name']}/{EXPERIMENT_TIMESTAMP}")
                
        self.config["config_paths"] = {
            "global_config": global_config_path,
            "benchmark_config": benchmark_config_path, 
            "experiment_config": experiment_config_path
        }
        
    def run(self): 
        pass
    
    def load_agent(self):
        
        if self.config["benchmark"] == "appworld": 
            self.agent_config = AppWorldAgentConfig(**self.config)
            self.agent_class = AppWorldAgent
        elif self.config["benchmark"] == "officebench": 
            self.agent_config = OfficeBenchAgentConfig(**self.config)
            self.agent_class = OfficeBenchAgent
        elif self.config["benchmark"] == "musique": 
            self.agent_config = {}
            self.agent_class = SmolagentsAgent 
        else: 
            raise NotImplementedError("Benchmark unknown")
        
    def load_benchmark_data(self): 
        raise NotImplementedError("The Runner needs to define a data loader.")
    
    def configure_environment(self):
        
        # Check input directory exists
        assert Path(self.config["input_path"]).exists(), "Input data path does not exist."         
        
        # Setup output directories
        self.config["output_path"].mkdir(parents=True, exists_ok=True)
        
        # Setup temporary workdir
        self.config["cachedir"] = Path(f"{self.config['output_path']}/cache")
        self.config["cachedir"].mkdir(parents=True, exists_ok=True) 
        
        # Create Agent environment
        
        pass
    
    def run_task(self, task: Dict):
        
        # Copy task to workdir
        self.copy_task_data_to_workdir()
        
        pass
    
    def cleanup(self):
       self.config["workdir"].rmdir() 
       
    def copy_task_data_to_workdir(): 
        raise NotImplementedError("The Runner needs to define a workdir copy fn.")
    
    
    
    