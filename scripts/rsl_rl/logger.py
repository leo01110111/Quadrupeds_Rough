import csv
import torch
import re
from typing import List, Tuple
from pathlib import Path


class EnvironmentLogger:
    """Logger for continuously recording observations and actions from a single environment to CSV."""
    
    def __init__(
        self,
        observation_labels: List[str],
        action_labels: List[str],
        filename: str = "data.csv"
    ):
        """
        Initialize the environment logger.
        
        Args:
            observation_labels (List[str]): Labels for each observation element with format "name(size)"
            action_labels (List[str]): Labels for each action element with format "name(size)"
            filename (str): Output CSV filename
        """
        self.observation_labels = observation_labels
        self.action_labels = action_labels
        self.filename = filename
        self.step_count = 0
        self.file_initialized = False
        
        # Parse group information from labels
        self.obs_groups = self._parse_labels(observation_labels)
        self.act_groups = self._parse_labels(action_labels)
        
        # Initialize the CSV file with headers
        self._initialize_file()
    
    def _parse_labels(self, labels: List[str]) -> List[Tuple[str, int]]:
        """
        Parse labels to extract group name and size.
        Format: "name(size)" -> ("name", size)
        
        Args:
            labels: List of labels with format "name(size)"
            
        Returns:
            List of tuples (name, size)
        """
        groups = []
        for label in labels:
            match = re.match(r'(\w+)\((\d+)\)', label)
            if match:
                name = match.group(1)
                size = int(match.group(2))
                groups.append((name, size))
            else:
                raise ValueError(f"Invalid label format: {label}. Expected format: 'name(size)'")
        return groups
    
    def _initialize_file(self):
        """Create the CSV file with headers."""
        # Simple headers for step and type
        headers = ["step", "type", "data"]
        
        with open(self.filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
        self.file_initialized = True
        print(f"CSV file '{self.filename}' initialized!")
    
    def _create_grouped_headers(self, groups: List[Tuple[str, int]]) -> List[str]:
        """
        Create grouped headers with 3 elements per group per row.
        
        Args:
            groups: List of (name, size) tuples
            
        Returns:
            List of header strings with group labels
        """
        headers = []
        for name, size in groups:
            # Calculate how many rows we need for this group (ceil division)
            num_rows = (size + 2) // 3  # +2 to round up
            for row_idx in range(num_rows):
                start_idx = row_idx * 3
                end_idx = min(start_idx + 3, size)
                indices = list(range(start_idx + 1, end_idx + 1))
                # Create subheader like "base_lin_vel(1-3)", "base_lin_vel(4-6)", etc.
                header = f"{name}({indices[0]}-{indices[-1]})"
                headers.append(header)
        return headers
    
    def _format_grouped_data_with_labels(self, data: List[float], groups: List[Tuple[str, int]]) -> str:
        """
        Format data into groups with labels, 3 elements per row, with newlines between rows.
        
        Args:
            data: Flat list of values
            groups: List of (name, size) tuples
            
        Returns:
            Formatted string with labels and values separated by newlines
        """
        lines = []
        idx = 0
        
        for name, size in groups:
            # Extract elements for this group
            group_data = data[idx:idx + size]
            idx += size
            
            # Chunk group data into rows of 3
            for i in range(0, len(group_data), 3):
                chunk = group_data[i:i+3]
                chunk_idx = i // 3
                start_pos = i + 1
                end_pos = min(i + 3, len(group_data)) + 1
                
                # Format as: "name(start-end): value1,value2,value3"
                label = f"{name}({start_pos}-{end_pos-1})"
                values = ",".join(str(v) for v in chunk)
                lines.append(f"{label}: {values}")
        
        return "\n".join(lines)
    
    def log_step(
        self,
        observation: torch.Tensor,
        action: torch.Tensor
    ):
        """
        Log a single step of observations and actions to the CSV file.
        Labels are repeated with values, 3 elements per row, separated by newlines.
        
        Args:
            observation (torch.Tensor): Observation tensor of shape (1, x) on GPU
            action (torch.Tensor): Action tensor of shape (1, x) on GPU
        """
        # Validate shapes - should be (1, x)
        assert observation.dim() == 2 and observation.shape[0] == 1, \
            f"Observation must have shape (1, x), got {observation.shape}"
        assert action.dim() == 2 and action.shape[0] == 1, \
            f"Action must have shape (1, x), got {action.shape}"
        
        # Extract dimensions
        obs_dim = observation.shape[1]
        act_dim = action.shape[1]
        
        # Validate total dimensions match group sizes
        obs_total = sum(size for _, size in self.obs_groups)
        act_total = sum(size for _, size in self.act_groups)
        assert obs_dim == obs_total, \
            f"Observation dimension mismatch: got {obs_dim}, expected {obs_total}"
        assert act_dim == act_total, \
            f"Action dimension mismatch: got {act_dim}, expected {act_total}"
        
        # Convert GPU tensors to CPU and then to list, rounded to 2 decimal places
        obs_list = [round(x, 2) for x in observation.squeeze(0).detach().cpu().numpy().tolist()]
        act_list = [round(x, 2) for x in action.squeeze(0).detach().cpu().numpy().tolist()]
        
        # Format observations and actions with labels and newlines
        obs_data = self._format_grouped_data_with_labels(obs_list, self.obs_groups)
        act_data = self._format_grouped_data_with_labels(act_list, self.act_groups)
        
        # Add newline before step, observation
        obs_data = f"\nStep {self.step_count}, Observation:\n{obs_data}"
        # Add newline before step, action
        act_data = f"\nStep {self.step_count}, Action:\n{act_data}"
        
        # Append observation row
        with open(self.filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
            writer.writerow([obs_data])
        
        # Append action row
        with open(self.filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
            writer.writerow([act_data])
        
        self.step_count += 1
    
    def get_step_count(self) -> int:
        """Get the current step count."""
        return self.step_count
    
    def reset(self):
        """Reset the logger and clear the file."""
        self.step_count = 0
        self._initialize_file()
        print(f"Logger reset! Step count: {self.step_count}")


if __name__ == "__main__":
    # Example usage with grouped labels
    obs_labels = ["base_lin_vel(3)", "base_ang_vel(3)", "proj_grav(3)", "vel_cmd(3)", 
                  "joint_pos(12)", "joint_vel(12)", "action(12)"]
    act_labels = ["joint_pos(12)"]
    
    # Create logger instance
    logger = EnvironmentLogger(
        observation_labels=obs_labels,
        action_labels=act_labels,
        filename="data.csv"
    )
    
    # Simulate a loop logging 10 steps
    # Tensors should be on GPU and have shape (x, 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Total obs: 3+3+3+3+12+12+12 = 48
    # Total act: 12
    for step in range(10):
        obs = torch.randn(1, 48, device=device)
        act = torch.randn(1, 12, device=device)
        logger.log_step(obs, act)
    
    print(f"Logging complete! Total steps: {logger.get_step_count()}")
