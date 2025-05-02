import numpy as np
import plotly.express as px


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
        total_manager_hours = 0
        manager_count = 0
        for employee in self.all_employees:
            if isinstance(employee, Manager):
                team_meeting_duration = 0.5 + 0.1 * len(employee.team)
                total_meeting_hours += team_meeting_duration
                if employee.manager:
                    total_meeting_hours += 0.5
                    team_meeting_duration += 0.5
                total_manager_hours += team_meeting_duration
                manager_count += 1
        return total_meeting_hours, total_manager_hours, manager_count


# Run experiments and plot both heatmaps
def run_experiments():
    layers_range = range(1, 8)
    team_sizes = range(1, 10)

    emp_ratio_matrix = np.zeros((len(layers_range), len(team_sizes)))
    mgr_ratio_matrix = np.zeros((len(layers_range), len(team_sizes)))

    for i, layers in enumerate(layers_range):
        for j, team_size in enumerate(team_sizes):
            sim = CompanySimulator(num_layers=layers, team_size=team_size)
            total_hours, mgr_hours, mgr_count = sim.simulate_meetings()
            emp_ratio = total_hours / len(sim.all_employees)
            mgr_ratio = mgr_hours / mgr_count if mgr_count else 0

            emp_ratio_matrix[i, j] = emp_ratio
            mgr_ratio_matrix[i, j] = mgr_ratio

    # Heatmap 1: Per Employee
    fig1 = px.imshow(
        emp_ratio_matrix,
        labels=dict(x="Team Size", y="Management Layers", color="Hours / Employee"),
        x=list(team_sizes),
        y=list(layers_range),
        text_auto=".2f",
        title="Meeting Hours per Employee",
        aspect="auto",
        color_continuous_scale="YlGnBu",
    )
    fig1.update_layout(margin=dict(t=40, l=0, r=0, b=0))
    fig1.show()

    # Heatmap 2: Per Manager
    fig2 = px.imshow(
        mgr_ratio_matrix,
        labels=dict(x="Team Size", y="Management Layers", color="Hours / Manager"),
        x=list(team_sizes),
        y=list(layers_range),
        text_auto=".2f",
        title="Meeting Hours per Manager",
        aspect="auto",
        color_continuous_scale="OrRd",
    )
    fig2.update_layout(margin=dict(t=40, l=0, r=0, b=0))
    fig2.show()


# Main execution
if __name__ == "__main__":
    sim = CompanySimulator(num_layers=5, team_size=5)
    total, mgr_total, mgr_count = sim.simulate_meetings()
    print(f"Total Employees: {len(sim.all_employees)}")
    print(f"Total Meeting Hours: {total:.2f}")
    print(f"Per Employee: {total / len(sim.all_employees):.2f}")
    print(f"Per Manager: {mgr_total / mgr_count:.2f}")

    run_experiments()
