import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


# Base class for an employee
class Employee:
    def __init__(self, id, manager=None):
        self.id = id
        self.manager = manager


# Manager is a type of Employee that can have a team
class Manager(Employee):
    def __init__(self, id, manager=None):
        super().__init__(id, manager)
        self.team = []

    def add_direct_report(self, employee):
        self.team.append(employee)
        employee.manager = self


# Simulator class to model the company structure
class CompanySimulator:
    def __init__(self, num_layers, team_size):
        self.num_layers = num_layers
        self.team_size = team_size
        self.all_employees = []
        self.ceo = Manager("CEO")
        self.all_employees.append(self.ceo)
        self.build_company()

    def build_company(self):
        current_layer = [self.ceo]
        for layer in range(self.num_layers):
            next_layer = []
            for manager in current_layer:
                for _ in range(self.team_size):
                    if layer != self.num_layers - 1:
                        new_manager = Manager(f"Manager_{len(self.all_employees)}")
                        manager.add_direct_report(new_manager)
                        next_layer.append(new_manager)
                        self.all_employees.append(new_manager)
                    else:
                        new_employee = Employee(f"Employee_{len(self.all_employees)}")
                        manager.add_direct_report(new_employee)
                        self.all_employees.append(new_employee)
            current_layer = next_layer

    def simulate_meetings(self):
        total_meeting_hours = 0
        for employee in self.all_employees:
            if isinstance(employee, Manager):
                team_meeting_duration = 0.5 + 0.1 * len(employee.team)
                total_meeting_hours += team_meeting_duration
                if employee.manager:
                    total_meeting_hours += 0.5
        return total_meeting_hours


# Run simulations and generate both plots
def run_experiments():
    layers_range = range(1, 8)
    team_sizes = range(1, 10)
    results = []

    for layers in layers_range:
        for team_size in team_sizes:
            sim = CompanySimulator(num_layers=layers, team_size=team_size)
            meetings = sim.simulate_meetings()
            meeting_ratio = meetings / len(sim.all_employees)
            results.append((layers, team_size, meeting_ratio))

    # Prepare data for plotting
    xs = [r[0] for r in results]
    ys = [r[1] for r in results]
    zs = [r[2] for r in results]

    # 1. 3D Scatter Plot
    fig1 = plt.figure(figsize=(10, 6))
    ax = fig1.add_subplot(111, projection="3d")
    ax.scatter(xs, ys, zs, c=zs, cmap="viridis")
    ax.set_xlabel("Management Layers")
    ax.set_ylabel("Team Size")
    ax.set_zlabel("Meeting Hours per Employee")
    plt.title("3D View: Meeting Burden by Structure")
    plt.tight_layout()

    # 2. Heatmap Plot
    heatmap_data = np.zeros((len(layers_range), len(team_sizes)))
    for layer, team, ratio in results:
        heatmap_data[layer - 1, team - 1] = ratio

    fig2 = plt.figure(figsize=(10, 6))
    sns.heatmap(
        heatmap_data,
        annot=True,
        xticklabels=list(team_sizes),
        yticklabels=list(layers_range),
        cmap="YlGnBu",
    )
    plt.xlabel("Team Size")
    plt.ylabel("Management Layers")
    plt.title("Heatmap: Meeting Hours per Employee")

    plt.show()


# Main execution block
if __name__ == "__main__":
    sim = CompanySimulator(num_layers=5, team_size=5)
    meetings = sim.simulate_meetings()
    print(f"Total Employees: {len(sim.all_employees)}")
    print(f"Total Meeting Hours per Week: {meetings:.2f}")
    print(f"Meeting ratio: {meetings / len(sim.all_employees):.2f}")

    run_experiments()
