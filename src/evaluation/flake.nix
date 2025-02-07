{
  description = "Python dev shell with PyTorch (CUDA enabled) integrated";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = { allowUnfree = true; };
        };

        # Create a Python 3.11 environment that includes torch and numpy.
        pythonEnv = pkgs.python311.withPackages (ps: with ps; [
          pytorch-bin# Prebuilt PyTorch with CUDA support.
          numpy
        ]);
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pythonEnv
            pkgs.cudatoolkit  # Provides the necessary CUDA runtime libraries.
          ];

          shellHook = ''
            echo "Welcome to the Python shell with integrated PyTorch (CUDA enabled)!"
            export CUDA_PATH=${pkgs.cudatoolkit}
            # Ensure the system finds the NVIDIA libraries at runtime.
            export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [ pkgs.linuxPackages.nvidia_x11 ]}
          '';
        };
      }
    );
}
