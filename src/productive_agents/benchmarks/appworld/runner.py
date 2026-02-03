import click
import inspect
import json
import os 
import shutil
import wandb

from pathlib import Path


from productive_agents.agents.officebench.config import OfficeBenchAgentConfig
from productive_agents.agents.officebench.agent import OfficeBenchAgent
from productive_agents.env.officebench import OfficeBenchEnvConfig, OfficeBenchEnv
from productive_agents.benchmarks.base import BenchmarkRunner
from productive_agents.utils.logging import setup_logger

from typing import List, Dict


logger = setup_logger()


class OfficeBenchBenchmarkRunner(BenchmarkRunner):
    
    def __init__(self, global_config_path, benchmark_config_path, experiment_config_path, output_path = None):
        super().__init__(global_config_path, benchmark_config_path, experiment_config_path, output_path)
        
        # Task list
        self.task_list = sorted([i for i in self.config["input_path"].glob("*") if i.is_directory()])
    
    def load_benchmark_data(self):
        pass
    
    def copy_task_data_to_workdir(self, task_id: str):
        task_dir = Path(f"{self.config['cachedir']}/tasks/{task_id}") 
        if task_dir.exists() and self.config["cache_responses"]: 
            logger.warning("Task found in cache directory. Skipping...")
            return {"status": "cached"}
        elif task_dir.exists() and not self.config["cache_responses"]:
             logger.warning(f"'use_cached' set to False. Re-creating workdir for task {task_id}.")
             task_dir.rmdir()
        else: 
            task_dir.mkdir(parents=True)
             
        shutil.copytree(
            f"{self.config['input_path']}/tasks/{task_id}", 
            f"{self.config['cachedir']}/tasks/{task_id}", 
            dirs_exist_ok=True, 
            ignore=shutil.ignore_patterns('*.pyc', '__pycache__') 
        )
        
        logger.info(f"Data prepared for task {task_id}.")
        
    def get_configuration(self, task_id): 
        
        # Experiment config
        required_exp_params = inspect(OfficeBenchAgentConfig.__init__)
        relevant_exp_params = {k: v for k, v in self.config.items() if k in required_exp_params} 
        experiment_config = OfficeBenchAgentConfig(**required_exp_params)
        
        # Environment config 
        required_env_params = inspect(OfficeBenchAgentConfig.__init__)
        relevant_env_params = {k: v for k, v in self.config.items() if k in required_env_params}
        environment_config = OfficeBenchEnvConfig(**relevant_env_params) 
        
        logger.info(f"Configuration for task {task_id} loaded and prepared.")
        return experiment_config, environment_config
        
    def setup_experiment_env(self):
        self.env = OfficeBenchEnv(config=self.environment_config)
        self.env.reset()
        
    def setup_agent(self, task_config: Dict):
        required_agent_params = inspect(OfficeBenchAgent.__init__) 
        relevant_agent_params = {k: v for k, v in self.config.items() if k in required_agent_params}
        self.agent = OfficeBenchAgent(
            env=self.env,
            task_config=task_config,
            exp_config=self.experiment_config, 
            **relevant_agent_params
        )
        
    def run_task(self, task_id):
        subtask_list = [i for i in Path(f"{self.config['input_dir']}/{task_id}/subtasks") if i.endswith(".json")]
        logger.info(f"Sub-tasks captured and prepared for processing (scope: {task_id})")
        for subtask_file_name in subtask_list:
            experiment_config, environment_config = self.get_configuration(task_id=task_id)
            subtask_id = subtask_file_name.split("/")[-1].split(".")[0]
            subtask_output_dir = Path(f"{self.config['output_path']}/{task_id}/{subtask_id}")
            subtask_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Task caching 
            llm_cache = None
            if Path(f"{subtask_output_dir}/llm_history.json").exists() and self.config["cache_responses"]: 
                logger.info(f"Using cached responses (total messages: {len(llm_cache.keys())}, task {task_id})")
                llm_cache = {}
                llm_history = json.load(open(f'{subtask_output_dir}/llm_history.json'))
                for item in llm_history:
                    llm_cache[item[1]] = item[2]
                
            
            
                    
    def run_benchmark(self):
        
        run_name = ""
        task_list = []
        
        with wandb.init(
            entity=self.config["wandb_entity"],
            project=self.config["wandb_project"],
            name=run_name,
            config=self.config
        ) as run:
            for task_id in self.task_list:
                stats = self.run_task(task_id=task_id)
                
                
@click.command()     
@click.option("-g", "--global-config", help="Path to the global configuration yaml file.")
@click.option("-b", "--benchmark-config", help="Path to the benchmark-specific configuration yaml file.")
@click.option("-e", "--experiment-config", help="Path to the experiment-specific configuration yaml file.")
@click.option("-o", "--output-path", help="Path to the experiment-specific configuration yaml file.")
def launch(global_config, benchmark_config, experiment_config, output_path):
    
    runner = OfficeBenchBenchmarkRunner(
        global_config_path=global_config,
        benchmark_config_path=benchmark_config,
        experiment_config_path=experiment_config,
        output_path=output_path
    )
    
    runner.run_benchmark()
           
                
if __name__ == "__main__":
    launch()
