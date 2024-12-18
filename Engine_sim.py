import numpy as np
import matplotlib.pyplot as plt
import subprocess


def comment_lines_in_file(filename, line_indices):
    """
    Reads a .inp file, adds a '!' at the beginning of specified lines, and writes back the changes.

    :param filename: Path to the .inp file.
    :param line_indices: List of line indices (0-based) to comment.
    """
    with open(filename, "r") as file:
        lines = file.readlines()

    for index in line_indices:
        if 0 <= index < len(lines):
            # Add '!' only if it's not already there
            if not lines[index].strip().startswith("!"):
                lines[index] = f"!{lines[index]}"

    with open(filename, "w") as file:
        file.writelines(lines)

def uncomment_lines_in_file(filename, line_indices):
    """
    Reads a .inp file, removes a '!' at the beginning of specified lines, and writes back the changes.

    :param filename: Path to the .inp file.
    :param line_indices: List of line indices (0-based) to comment.
    """
    with open(filename, "r") as file:
        lines = file.readlines()

    for index in line_indices:
        if 0 <= index < len(lines):
            # Remove '!' only if it starts the line
            if lines[index].strip().startswith("!"):
                lines[index] = lines[index].lstrip("!")  # Remove leading '!' only

    with open(filename, "w") as file:
        file.writelines(lines)


class EngineMechanism:
    def __init__(self, RPM, theta2):
        # Mechanism parameters
        self.RPM = RPM
        self.theta2 = theta2

        # Geometry
        self.nc = 4  # number of cylinders
        self.s = 400e-3  # stroke length [m]
        self.b = 320e-3  # bore diameter [m]
        self.epsilon = 1.6 / 2  # ratio of crankshaft radius to connecting rod length
        self.theta_init = -180
        self.theta_end = 160

        self.l = 2 * self.epsilon * self.s  # connecting rod length [m]
        self.Vd = (np.pi / 4) * self.b**2 * self.s  # Volume of 1 cylinder
        self.Vtdc = self.Vd / (12 - 1)  # V top dead center, with compression ratio r = 12
        self.R = 2 * self.l / self.s  # geometric quantity

        # Thermodynamics
        self.r = 12  # compression ratio
        self.Ti = 373  # intake temperature [K]
        self.Patm = 1  # atmospheric pressure [atm]
        self.polytropic = 1.35
        self.gamma = 1.26

    def vol(self, theta):
        return self.Vtdc * (
            1 + (self.r - 1) / 2 * (1 - np.cos(np.radians(theta))) +
            1 / (2 * self.epsilon) * (1 - np.sqrt(1 - self.epsilon**2 * np.sin(np.radians(theta))**2))
        )

    def compression(self, theta1, theta2, P1, T1):
        V1 = self.vol(theta1)
        V2 = self.vol(theta2)
        V = np.linspace(V1, V2, 100)
        P = P1 * (V / V1) ** -self.polytropic
        T = T1 * (P / P1) ** ((self.polytropic - 1) / self.polytropic)
        return P, V, T

    def expansion(self, theta1, theta2, P1, T1):
        V1 = self.vol(theta1)
        V2 = self.vol(theta2)
        V = np.linspace(V1, V2, 100)
        P = P1 * (V / V1) ** -self.polytropic
        T = T1 * (P / P1) ** ((self.polytropic - 1) / self.polytropic)
        return P, V, T

    def file_handling(self, filename, parameter, new_value):
        with open(filename, "r") as file:
            lines = file.readlines()

        for i, line in enumerate(lines):
            if parameter in line:
                lines[i] = f"{parameter} {new_value:.2f}\n"
                break

        with open(filename, "w") as file:
            file.writelines(lines)

    def read_dot_out_file(self, filename):
        with open(filename, "r") as file:
            data = np.loadtxt(file)
        return data

    def execute_simulation(self,  input_filename, output_filename):
        # Compression
        P2, V2, T2 = self.compression(self.theta_init, self.theta2, self.Patm, self.Ti)
        p2, v2, t2 = P2[-1], V2[-1], T2[-1]

        # Update input file
        input_path = f"{input_filename}"
        self.file_handling(input_path, "TEMP", t2)
        self.file_handling(input_path, "PRES", p2)

        # Run external executables
        subprocess.run(["ckinterp.exe"])
        subprocess.run(["senkin.exe"])

        # Read output
        output_path = f"{output_filename}"
        data = self.read_dot_out_file(output_path)
        time, P3, T3 = data[:, 0], data[:, 1], data[:, 2]
        return P2, V2, T2, time, P3, T3


    def run_full_simulation(self, input_filename = "senkin.inp", output_filename = "senkin.out"):
        P2, V2, T2, time, P3, T3 = self.execute_simulation(input_filename, output_filename)
        p3max, t3max = max(P3), max(T3)
        p3ind = np.argmax(P3)

        Delta_theta = np.degrees(2 * np.pi * self.RPM / 60) * time[p3ind]
        theta3 = self.theta2 + Delta_theta
        v3 = self.vol(theta3)
        V3 = np.linspace(V2[-1], v3, len(P3))

        if theta3 < 0:
            P34, V34, T34 = self.compression(theta3, 0, p3max, t3max)
            P4, V4, T4 = self.expansion(0, self.theta_end, P34[-1], T34[-1])
            Pmax, Tmax = P34[-1], T34[-1]
            penalty = True
        else:
            P4, V4, T4 = self.expansion(theta3, self.theta_end, p3max, t3max)
            Pmax, Tmax = p3max, t3max
            penalty = False

        return Pmax, Tmax, penalty, theta3
    
    def plot_results(self, Ptot, Vtot, time, P3, T3):
        plt.figure(1)
        plt.plot(time, P3)
        plt.title("Combustion")
        plt.xlabel("time [sec]")
        plt.ylabel("Pressure [bar]")
        plt.show()

        plt.figure(2)
        plt.plot(Vtot, Ptot)
        plt.title("Engine cycle")
        plt.xlabel(r"V [$m^3$]")
        plt.ylabel("P [bar]")
        plt.show()

        plt.figure(3)
        plt.plot(time, T3)
        plt.title("Combustion")
        plt.xlabel("time [sec]")
        plt.ylabel("Temperature [K]")
        plt.show()




if __name__ == "__main__":
    # Parameters
    RPM, theta2 = [77.14665368,  -0.741]
    full_mechanism = True

    if full_mechanism:
        indcs = []
    else:
        indcs = [112,114,115,127,128,141,142,143,147,149,150]

    indcs = [x-1 for x in indcs]
    # for i in indcs:
    comment_lines_in_file(filename = "chem.inp", line_indices = indcs)
    engine = EngineMechanism(RPM, theta2)
    input_filename = "senkin.inp"
    output_filename = "senkin.out"

    P2, V2, T2, time, P3, T3 = engine.execute_simulation(input_filename, output_filename)

    p3max, t3max = max(P3), max(T3)
    p3ind = np.argmax(P3)

    Delta_theta = np.degrees(2 * np.pi * RPM / 60) * time[p3ind]
    theta3 = theta2 + Delta_theta
    v3 = engine.vol(theta3)
    V3 = np.linspace(V2[-1], v3, len(P3))

    if theta3 < 0:
        P34, V34, T34 = engine.compression(theta3, 0, p3max, t3max)
        P4, V4, T4 = engine.expansion(0, engine.theta_end, P34[-1], T34[-1])
        Pmax, Tmax = P34[-1], T34[-1]
    else:
        P4, V4, T4 = engine.expansion(theta3, engine.theta_end, p3max, t3max)
        Pmax, Tmax = p3max, t3max

    Ptot = np.concatenate([P2, P3, P4])
    Ttot = np.concatenate([T2, T3, T4])
    Vtot = np.concatenate([V2, V3, V4])

    engine.plot_results(Ptot, Vtot, time, P3, T3)

 
    uncomment_lines_in_file(filename = "chem.inp", line_indices = indcs)
    print(f"Pmax: {Pmax}")
    print(f"Tmax: {Tmax}")
