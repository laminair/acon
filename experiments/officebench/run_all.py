#!/usr/bin/env python3
"""
Simplified entry point for running all OfficeBench experiments.

This script runs all tasks from the tasks folder with minimal configuration overhead.
"""

import argparse
import shutil
import os
import json
import time
import logging
import wandb
import yaml

from run import main
from experiments.officebench.evaluation import task_discovery
from experiments.officebench.evaluation.task_evaluator import TaskEvaluator
from experiments.officebench.evaluation.main import load_non_image_tasks


from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from pathlib import Path


EXPERIMENT_FOLDER_PATH = Path(__file__).parent

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Suppress noisy loggers
    for logger_name in ["httpx", "azure.identity", "azure.core"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def load_base_config(tag: str, model_name: str, base_config_path: str, use_workflow_memory: bool = False, co_config: dict = None) -> dict:
    """Load and create minimal configuration."""
    base_config_path = Path(base_config_path)
    
    if base_config_path.exists():
        with open(base_config_path) as f:
            config = yaml.safe_load(f)
    else:
        # Minimal fallback config
        config = {}
    
    # Override with runtime values
    config.update({
        'exp_id': f"{tag}_{model_name}",
        'model_name': model_name,
        'tag': tag,
        'debug_mode': True,
        'max_iter': 50,
        'use_workflow_memory': use_workflow_memory,
        'use_thinking_tokens': True,
        'prompt_file': f"{EXPERIMENT_FOLDER_PATH}/prompts/prompts_v3.json",
        'co_config': co_config
    })
    
    return config


def load_task_ids_from_split(split: str) -> list:
    """
    Load task IDs for a given dataset split.
    
    Args:
        split: Dataset split ('train' or 'test')
        
    Returns:
        List of task IDs for the split
    """
    split_file = f"{EXPERIMENT_FOLDER_PATH}/{split}_tasks.txt"
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file {split_file} not found. Please run split_tasks.py first.")
    
    with open(split_file, 'r') as f:
        task_ids = [line.strip() for line in f if line.strip()]
    
    return task_ids


def _process_task(i: int, task: str):
    if '-' not in task:
        return
    print(f'\n{"="*60}')
    print(f'Running task {task} ({i+1}/{len(task_list)}) from {args.split} split')
    print(f'{"="*60}')
    if not args.debug and os.path.exists(os.path.join(output_root_dir, task)):
        print(f"Output directory for task {task} already exists, skipping...")
        return
    subtask_list = os.listdir(f'{args.tasks_dir_name}/{task}/subtasks')
    task_success = True
    for subtask in subtask_list:
        print(f'Running subtask {subtask}')
        task_config_file = f'{args.tasks_dir_name}/{task}/subtasks/{subtask}'
        subtask_name = subtask.split('.')[0]
        output_folder = f'{args.tasks_dir_name}/{task}/outputs/{subtask_name}/{args.model_name}_{args.tag}'
        if os.path.exists(output_folder):
            print('Output folder exists, skipping')
            continue
        _output_dir = os.path.join(output_root_dir, f'{task}/{subtask_name}')
        try:
            main(
                task_dir=f'{args.tasks_dir_name}/{task}',
                task_config_file=task_config_file,
                output_dir=_output_dir,
                task=task,
                split=args.split,
                mode='force_new',
                exp_config=exp_config,
                model_name=args.model_name,
                debug_mode=args.debug,
                lora_name=args.lora_name,
            )
            print(f"✅ Subtask {subtask} completed successfully!")
        except Exception as e:
            print(f"❌ Subtask {subtask} failed: {str(e)}")
            task_success = False
            logging.exception(f"Exception in task {task}, subtask {subtask}")
            
    wandb.log({
        "task_success": 0 if task_success is False else 1
    })
    
    if task_success:
        successful_tasks.append(task)
        print(f"✅ Task {task} completed successfully!")
    else:
        failed_tasks.append(task)
        print(f"❌ Task {task} had failures")


def evaluate(
    model_name, 
    tag_name, 
    result_dir, 
    output_subdir, 
    debug:bool=False, 
    list_missing:bool=False, 
    html:bool=False,
    web:bool=False,
    task=None,
    only_text:bool=False,
    task_dir="tasks",
    split=None
):
    """
    Main function to evaluate tasks and generate reports.
    This function is taken from officebench/evaluation/main.py
    
    Args:
        model_name: Name of the model to evaluate
        tag_name: Tag for the evaluation
        result_dir: Directory to store results
        output_subdir: Subdirectory for outputs
        debug: Enable debug logging
        list_missing: List missing result cases
        html: Generate HTML report
        web: Generate web report
        task: Specific task to evaluate (optional)
        only_text: Only evaluate non-image tasks (text-only tasks)
        task_dir: Directory containing task definitions
        split: Split name to evaluate (e.g., 'train', 'test') for fold-based evaluation
    """
        
    # Discover tasks
    task_discovery_obj = task_discovery.TaskDiscovery()
    all_tasks_info = task_discovery_obj.discover_tasks(task_dir)
    
    # Filter tasks if only_text flag is set
    if only_text:
        non_image_tasks = load_non_image_tasks()
        all_tasks_info = [(task_id, subtask_id) for task_id, subtask_id in all_tasks_info 
                         if task_id in non_image_tasks]
        logging.info(f"Filtered to {len(all_tasks_info)} non-image tasks")
    
    # Initialize the evaluator
    evaluator = TaskEvaluator(model_name, tag_name, output_subdir, debug, split)
        
    # Generate statistics
    stats = evaluator.generate_stats()

    # Augment stats with explicit accuracy percentages so they are persisted to JSON
    try:
        for num_app_tag, tag_stats in stats.get('app_tags', {}).items():
            denom = tag_stats.get('result_available_avg', 0.0)
            tag_stats['accuracy_pct'] = (tag_stats['success_avg'] / denom * 100.0) if denom else 0.0
        overall_stats = stats.get('overall', {})
        denom_overall = overall_stats.get('result_available_avg', 0.0)
        overall_stats['accuracy_pct'] = (overall_stats['success_avg'] / denom_overall * 100.0) if denom_overall else 0.0
    except Exception as e:
        logging.error(f"Failed computing accuracy_pct fields: {e}")
        
    wandb.summary.update(stats)
    

def run_experiments(args): 
    with open(args.global_config, 'r') as f:
        global_config = yaml.safe_load(f)
        f.close() 

    # Load co_config if provided
    co_config = None
    if args.co_config_path and os.path.exists(args.co_config_path):
        with open(args.co_config_path, "r") as file:
            co_config = yaml.safe_load(file)

    # Create minimal configuration
    exp_config = load_base_config(
        tag=args.tag,
        base_config_path=args.appworld_config,
        model_name=args.model_name,
        use_workflow_memory=args.use_workflow_memory,
        co_config=co_config
    )

    config_to_log = global_config
    config_to_log.update(exp_config)
    config_to_log.update(co_config)

    # Clear the error log
    if os.path.exists("error.log"):
        os.remove("error.log")

    # Get task list based on split
    if args.task is not None:
        # Single task specified - validate it exists in filesystem
        available_tasks = sorted(os.listdir(args.tasks_dir_name))
        if args.task not in available_tasks:
            raise ValueError(f"Task {args.task} not found in {args.tasks_dir_name} folder")
        task_list = [args.task]
        print(f"Running single task: {args.task}")
    else:
        # Load tasks from split file
        task_list = load_task_ids_from_split(args.split)
        print(f"Running all tasks from {args.split} split: {len(task_list)} tasks")

    output_root_dir = f'{EXPERIMENT_FOLDER_PATH}/outputs/{args.model_name.replace("/","_")}_{args.tag}/{args.split}/'

    start_time = time.time()
    successful_tasks = []
    failed_tasks = []
    
    if global_config["wandb"]["enable"] is True and not args.debug:
        run = wandb.init(
            entity=global_config["wandb"]["entity"],
            project=global_config["wandb"]["project_name"], 
            config=config_to_log
        )
    else: 
        run = None

    if args.debug:
        for i, task in enumerate(task_list):
            _process_task(i, task)
            break  # preserve original behavior of single task in debug
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn('[progress.description]{task.description}'),
            BarColumn(),
            TextColumn('{task.completed}/{task.total}'),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=False,
        ) as progress:
            task_id = progress.add_task('OfficeBench Tasks', total=len(task_list))
            for i, task in enumerate(task_list):
                _process_task(i, task)
                progress.advance(task_id, 1)

    # Calculate and print total experiment time
    end_time = time.time()
    total_time = end_time - start_time

    # Convert to hours, minutes, and seconds
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    print(f"\n{'='*60}")
    print(f"OFFICEBENCH EXPERIMENT COMPLETED")
    print(f"{'='*60}")
    print(f"Split: {args.split}")
    print(f"Model: {args.model_name}")
    print(f"Tag: {args.tag}")
    print(f"Total tasks: {len(task_list)}")
    print(f"Successful: {len(successful_tasks)}")
    print(f"Failed: {len(failed_tasks)}")
    if len(task_list) > 0:
        print(f"Success rate: {len(successful_tasks)/len(task_list)*100:.1f}%")
    print(f"Total experiment time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"{'='*60}")

    if failed_tasks:
        print(f"Failed tasks: {failed_tasks}")

    # Save experiment summary
    summary = {
        'experiment_config': exp_config,
        'split': args.split,
        'total_tasks': len(task_list),
        'successful_tasks': successful_tasks,
        'failed_tasks': failed_tasks,
        'success_rate': len(successful_tasks)/len(task_list) if len(task_list) > 0 else 0,
        'total_time_seconds': total_time,
        'run_timestamp': time.time()
    }

    if run is not None: 
        wandb.summary.update(summary)

    summary_path = os.path.join(output_root_dir, 'experiment_summary.json')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Experiment summary saved to: {summary_path}")

    if args.output_dir is not None:
        # Copy outputs to specified directory
        target_dir = f'{args.output_dir}/{args.model_name.replace("/","_")}_{args.tag}/{args.split}/'
        shutil.copytree(output_root_dir, target_dir, dirs_exist_ok=True)
        print(f'Outputs copied to {target_dir}')
        
    # Compute final results
    evaluate(
        model_name=args.model_name, 
        tag_name=args.tag, 
        split=args.split
    )
    
    wandb.finish()
    print("Done!")



if __name__ == "__main__": 
    
    # Create the parser
    parser = argparse.ArgumentParser(description="Run OfficeBench experiments with train/test splits")
    parser.add_argument("--task", type=str, help="Task to run", default=None)
    parser.add_argument("--split", type=str, default="train", help="Dataset split to run")
    parser.add_argument("--tag", type=str, help="Tag for the experiment", default="debug")
    parser.add_argument("--model_name", type=str, help="Model name to use", default="gpt-4.1")
    parser.add_argument("--use_workflow_memory", action='store_true', help="Use workflow memory")
    parser.add_argument("--tasks_dir_name", type=str, help="tasks directory name", default="tasks")
    parser.add_argument("--output_dir", type=str, help="output directory", default=None)
    parser.add_argument("--co_config_path", type=str, help="Context optimization config file", default=None)
    parser.add_argument("--debug", action='store_true', help="Enable debug mode")
    parser.add_argument("--lora_name", type=str, help="LoRA model name for agent", default=None)
    parser.add_argument("--appworld-config", type=str, default=f"{EXPERIMENT_FOLDER_PATH}/configs/base_config.yaml", help="Appworld experiment base config file path")
    parser.add_argument("--global-config", type=str, default=f"{EXPERIMENT_FOLDER_PATH.parent.parent}configs/global_config.yaml", help="Global experiment config file path")

    # Parse the arguments
    args = parser.parse_args()

    setup_logging()
    run_experiments(args)
