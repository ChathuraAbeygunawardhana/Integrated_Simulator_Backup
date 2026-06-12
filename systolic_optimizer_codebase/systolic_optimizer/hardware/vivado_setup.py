#!/usr/bin/env python3
"""
Vivado xsim Integration for Systolic Array Simulation
======================================================
Provides Python interface to run simulations in Vivado xsim
"""

import os
import subprocess
import json
from pathlib import Path
from typing import Optional, Dict, List


class VivadoSimulator:
    """Interface to Vivado xsim for systolic array simulation."""

    def __init__(
        self,
        vivado_path: Optional[str] = None,
        project_name: str = "systolic_array_sim",
        fpga_part: str = "xc7z020clg400-1",
    ):
        """
        Initialize Vivado simulator.

        Args:
            vivado_path: Path to Vivado executable (auto-detect if None)
            project_name: Name of Vivado project
            fpga_part: FPGA part number (default: Zynq-7000)
        """
        self.vivado_path = vivado_path or self._find_vivado()
        self.project_name = project_name
        self.fpga_part = fpga_part
        self.project_dir = Path(f"vivado_project_{project_name}")
        self.xsim_dir = Path("xsim_work")
        self.verified = False

    def _find_vivado(self) -> str:
        """Auto-detect Vivado installation."""
        try:
            result = subprocess.run(
                ["where", "vivado"], capture_output=True, text=True  # Windows
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass

        try:
            result = subprocess.run(
                ["which", "vivado"], capture_output=True, text=True  # Linux/Mac
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass

        raise RuntimeError(
            "Vivado not found in PATH. Please install Vivado or set VIVADO_PATH"
        )

    def verify_installation(self) -> bool:
        """Verify Vivado is installed and accessible."""
        try:
            result = subprocess.run(
                [self.vivado_path, "-version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                print(f"✓ Vivado found: {result.stdout.split(chr(10))[0]}")
                self.verified = True
                return True
        except Exception as e:
            print(f"✗ Vivado verification failed: {e}")
        return False

    def create_project(
        self, verilog_sources: List[str], testbench_files: List[str]
    ) -> bool:
        """Create Vivado project with sources."""
        if not self.verified:
            print("Vivado not verified. Run verify_installation() first.")
            return False

        self.project_dir.mkdir(exist_ok=True)

        # Create TCL script for project creation
        tcl_script = f"""
create_project -force {self.project_name} {self.project_dir} -part {self.fpga_part}

# Add design sources
set_property CONSTRSET {{}} [current_fileset]
"""

        for src in verilog_sources:
            tcl_script += f"add_files -fileset sources_1 {src}\n"

        for tb in testbench_files:
            tcl_script += f"add_files -fileset sim_1 {tb}\n"

        tcl_script += f"""
set_property top systolic_spad_top [get_filesets sim_1]
set_property default_lib work [current_project]

# Save and close
save_project_as {self.project_name}
close_project
"""

        tcl_file = self.project_dir / "create_project.tcl"
        with open(tcl_file, "w") as f:
            f.write(tcl_script)

        # Run Vivado in batch mode
        result = subprocess.run(
            [
                self.vivado_path,
                "-mode",
                "batch",
                "-source",
                str(tcl_file),
                "-log",
                str(self.project_dir / "vivado_create.log"),
                "-journal",
                str(self.project_dir / "vivado_create.jou"),
            ],
            capture_output=False,
            timeout=300,
        )

        if result.returncode == 0:
            print(f"✓ Project created: {self.project_dir}")
            return True
        else:
            print(f"✗ Project creation failed")
            return False

    def compile_sources(self, verilog_sources: List[str]) -> bool:
        """Compile Verilog sources with xvlog."""
        self.xsim_dir.mkdir(exist_ok=True)

        try:
            for src in verilog_sources:
                cmd = [
                    "xvlog",
                    "-work",
                    str(self.xsim_dir),
                    "-sv",
                    "-d",
                    "SIM_MODE",
                    src,
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"✗ Failed to compile {src}")
                    print(result.stderr)
                    return False
                print(f"✓ Compiled {src}")

            return True
        except FileNotFoundError:
            print("✗ xvlog not found. Make sure Vivado is in PATH.")
            return False

    def elaborate(self, top_module: str = "systolic_spad_top") -> bool:
        """Elaborate design with xelab."""
        try:
            cmd = [
                "xelab",
                "-work",
                str(self.xsim_dir),
                "-debug",
                "all",
                "-timescale",
                "1ps/1ps",
                f"work.{top_module}",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if result.returncode == 0:
                print(f"✓ Elaborated {top_module}")
                return True
            else:
                print(f"✗ Elaboration failed")
                print(result.stderr)
                return False
        except FileNotFoundError:
            print("✗ xelab not found. Make sure Vivado is in PATH.")
            return False

    def run_simulation(
        self,
        top_module: str = "systolic_spad_top",
        gui: bool = False,
        timeout: int = 300,
    ) -> bool:
        """Run simulation with xsim."""
        try:
            cmd = ["xsim", f"{self.xsim_dir}/{top_module}"]

            if gui:
                cmd.append("-gui")
                print("Starting xsim GUI...")
            else:
                cmd.append("-R")
                cmd.extend(["-log", "xsim_results.log"])

            result = subprocess.run(
                cmd, capture_output=not gui, text=True, timeout=timeout
            )

            if result.returncode == 0:
                print("✓ Simulation completed successfully")
                return True
            else:
                print("✗ Simulation failed")
                if not gui:
                    print(result.stderr)
                return False
        except FileNotFoundError:
            print("✗ xsim not found. Make sure Vivado is in PATH.")
            return False
        except subprocess.TimeoutExpired:
            print(f"✗ Simulation timeout (>{timeout}s)")
            return False

    def run_full_flow(
        self, verilog_sources: List[str], testbench_files: List[str], gui: bool = False
    ) -> bool:
        """Run complete simulation flow."""
        print("\n" + "=" * 70)
        print("Vivado xsim Simulation Flow")
        print("=" * 70)

        # Verify
        print("\n1. Verifying Vivado installation...")
        if not self.verify_installation():
            return False

        # Compile
        print("\n2. Compiling sources...")
        if not self.compile_sources(verilog_sources + testbench_files):
            return False

        # Elaborate
        print("\n3. Elaborating design...")
        if not self.elaborate():
            return False

        # Simulate
        print("\n4. Running simulation...")
        if not self.run_simulation(gui=gui):
            return False

        print("\n" + "=" * 70)
        print("✓ Simulation flow completed successfully")
        print("=" * 70)
        return True


def main():
    """Example usage of VivadoSimulator."""

    # Define sources
    verilog_sources = [
        "systolic_spad_top.sv",
        "scratchpad_ctrl.sv",
        "scratchpad.sv",
        "axi_mem_bridge.sv",
        "page_table.v",
        "weight_stationary_top.sv",
        "weight_stationary_pkg.sv",
        "processing_element_ws.sv",
    ]

    testbench_files = [
        "systolic_spad_tb.sv",
    ]

    # Create simulator
    sim = VivadoSimulator()

    # Run simulation flow
    success = sim.run_full_flow(
        verilog_sources=verilog_sources,
        testbench_files=testbench_files,
        gui=False,  # Set to True to open GUI
    )

    if not success:
        exit(1)


if __name__ == "__main__":
    main()
