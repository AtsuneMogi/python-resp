let
  flake = builtins.getFlake (toString ./.);
  system = builtins.currentSystem;
in
# Option A: If you are using the modern 'devShells' standard
flake.devShells.${system}.default or flake.defaultApp.${system}

# Option B (Fallback): If your flake defines it as 'devShell' (singular)
# flake.devShell.${system}
