
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# =============================================================================
# PART 1: NEURAL NETWORK ARCHITECTURE
# =============================================================================
class JefferyHamelPINN(nn.Module):
    def __init__(self):
        super(JefferyHamelPINN, self).__init__()
        self.shared_layer1 = nn.Linear(1, 60)
        self.shared_layer2 = nn.Linear(60, 60)
        self.shared_layer3 = nn.Linear(60, 60)

        self.output_F = nn.Linear(60, 1)   # Velocity
        self.output_beta = nn.Linear(60, 1) # Temperature
        self.output_phi = nn.Linear(60, 1)  # Concentration

    def forward(self, x):
        x = torch.tanh(self.shared_layer1(x))
        x = torch.tanh(self.shared_layer2(x))
        x = torch.tanh(self.shared_layer3(x))

        F = self.output_F(x)      # Velocity profile
        beta = self.output_beta(x) # Temperature profile
        phi = self.output_phi(x)   # Concentration profile

        return F, beta, phi

# =============================================================================
# PART 2: SOLVER WITH PHASED OPTIMIZATION
# =============================================================================
class CompleteJefferyHamelSolver:
    def __init__(self):
        # Fixed parameters from paper: α=5°, Re=10
        self.alpha = np.radians(5)  # 5 degrees in radians
        self.Re = 10.0

        # Additional parameters for equations 9 & 10
        self.Pr = 0.7      # Prandtl number (air)
        self.Ec = 0.3      # Eckert number (moderate heating)
        self.Df = 0.5      # Dufour number
        self.Sc = 0.62     # Schmidt number (water vapor)
        self.Sr = 0.2      # Soret number

        self.model = JefferyHamelPINN()

        # Both optimizers
        self.adam_optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # LBFGS optimizer with tuned parameters
        self.lbfgs_optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            lr=0.1,           # Higher learning rate for LBFGS
            max_iter=20,      # Internal iterations per LBFGS step
            max_eval=25,      # Function evaluations
            tolerance_grad=1e-8,
            tolerance_change=1e-10,
            history_size=50,
            line_search_fn='strong_wolfe'  # Strong Wolfe line search
        )

        # Training variables
        self.current_collocation_points = None
        self.lbfgs_epoch_counter = 0
        self.optimizer_used = []  # Track which optimizer was used each epoch

        print("COMPLETE JEFFERY-HAMEL SOLVER INITIALIZED")
        print(f"α = {np.degrees(self.alpha):.1f}°, Re = {self.Re}")
        print(f"Pr = {self.Pr}, Ec = {self.Ec}, Sc = {self.Sc}")
        print(f"Df = {self.Df}, Sr = {self.Sr}")
        print("\nPHASED OPTIMIZATION STRATEGY:")
        print(f"✓ Phase 1: 2000 epochs with Adam optimizer")
        print(f"✓ Phase 2: 200 epochs with LBFGS optimizer")
        print(f"✓ Total epochs: 2200")

    # =============================================================================
    # PART 3: EQUATION 8 - MOMENTUM LOSS
    # =============================================================================
    def momentum_loss(self, x):
        """Equation 8: F′′′ + 2αRe·F·F′ + 4α²F′ = 0"""
        x.requires_grad_(True)
        F, _, _ = self.model(x)

        F_x = grad(F, x, torch.ones_like(F), create_graph=True)[0]
        F_xx = grad(F_x, x, torch.ones_like(F_x), create_graph=True)[0]
        F_xxx = grad(F_xx, x, torch.ones_like(F_xx), create_graph=True)[0]

        momentum_eq = F_xxx + 2 * self.alpha * self.Re * F * F_x + 4 * self.alpha**2 * F_x
        return torch.mean(momentum_eq**2)

    # =============================================================================
    # PART 4: EQUATION 9 - ENERGY LOSS
    # =============================================================================
    def energy_loss(self, x):
        """Equation 9: β′′ + 2α²(2 + (Pr Re/α)F)β + Pr Ec[4α²F² + (F′)²] + Pr Df[ϕ′′ + 4α²ϕ] = 0"""
        x.requires_grad_(True)
        F, beta, phi = self.model(x)

        F_x = grad(F, x, torch.ones_like(F), create_graph=True)[0]
        beta_x = grad(beta, x, torch.ones_like(beta), create_graph=True)[0]
        beta_xx = grad(beta_x, x, torch.ones_like(beta_x), create_graph=True)[0]
        phi_x = grad(phi, x, torch.ones_like(phi), create_graph=True)[0]
        phi_xx = grad(phi_x, x, torch.ones_like(phi_x), create_graph=True)[0]

        term1 = beta_xx
        term2 = 2 * self.alpha**2 * (2 + (self.Pr * self.Re / self.alpha) * F) * beta
        term3 = self.Pr * self.Ec * (4 * self.alpha**2 * F**2 + F_x**2)
        term4 = self.Pr * self.Df * (phi_xx + 4 * self.alpha**2 * phi)

        energy_eq = term1 + term2 + term3 + term4
        return torch.mean(energy_eq**2)

    # =============================================================================
    # PART 5: EQUATION 10 - MASS TRANSPORT LOSS
    # =============================================================================
    def mass_transport_loss(self, x):
        """Equation 10: ϕ′′ + 2α²(2 + (Re Sc/α)F)ϕ + Sc Sr[β′′ + 4α²β] = 0"""
        x.requires_grad_(True)
        F, beta, phi = self.model(x)

        phi_x = grad(phi, x, torch.ones_like(phi), create_graph=True)[0]
        phi_xx = grad(phi_x, x, torch.ones_like(phi_x), create_graph=True)[0]
        beta_x = grad(beta, x, torch.ones_like(beta), create_graph=True)[0]
        beta_xx = grad(beta_x, x, torch.ones_like(beta_x), create_graph=True)[0]

        term1 = phi_xx
        term2 = 2 * self.alpha**2 * (2 + (self.Re * self.Sc / self.alpha) * F) * phi
        term3 = self.Sc * self.Sr * (beta_xx + 4 * self.alpha**2 * beta)

        mass_eq = term1 + term2 + term3
        return torch.mean(mass_eq**2)

    # =============================================================================
    # PART 6: BOUNDARY CONDITIONS
    # =============================================================================
    def boundary_loss(self):
        """Boundary conditions for all three variables"""
        losses = []

        x0 = torch.tensor([[0.0]], dtype=torch.float32)
        x1 = torch.tensor([[1.0]], dtype=torch.float32)

        F0, beta0, phi0 = self.model(x0)
        F1, beta1, phi1 = self.model(x1)

        losses.append((F0 - 1.0)**2)
        losses.append((F1 - 0.0)**2)
        losses.append((beta0 - 1.0)**2)
        losses.append((beta1 - 1.0)**2)
        losses.append((phi0 - 1.0)**2)
        losses.append((phi1 - 1.0)**2)

        x0.requires_grad_(True)
        F0_grad, _, _ = self.model(x0)
        F0_x = grad(F0_grad, x0, torch.ones_like(F0_grad), create_graph=True)[0]
        losses.append(F0_x**2)

        return torch.mean(torch.stack(losses))

    # =============================================================================
    # PART 7: LBFGS CLOSURE FUNCTION
    # =============================================================================
    def lbfgs_closure(self):
        """Closure function required by LBFGS optimizer"""
        if self.current_collocation_points is None:
            return torch.tensor(0.0)

        self.lbfgs_optimizer.zero_grad()

        # Compute losses
        L_momentum = self.momentum_loss(self.current_collocation_points)
        L_energy = self.energy_loss(self.current_collocation_points)
        L_mass = self.mass_transport_loss(self.current_collocation_points)
        L_boundary = self.boundary_loss()

        # Total loss (same weights as Adam)
        total_loss = (L_momentum + L_energy + L_mass + 100 * L_boundary)

        total_loss.backward()
        return total_loss

    # =============================================================================
    # PART 8: PHASED TRAINING LOOP WITHOUT EARLY STOPPING
    # =============================================================================
    def train(self):
        """Train using Adam for first 2000 epochs, then LBFGS for 200 epochs"""
        print("\nStarting PHASED OPTIMIZATION Training...")
        print("NO EARLY STOPPING - Fixed epoch counts:")
        print(f"✓ Phase 1: 2000 epochs with Adam optimizer")
        print(f"✓ Phase 2: 200 epochs with LBFGS optimizer")
        print(f"✓ Total epochs: 2200")
        print("-" * 75)

        losses_history = {
            'total': [], 'momentum': [], 'energy': [],
            'mass': [], 'boundary': [], 'optimizer': []
        }

        # Fixed collocation points for LBFGS phase (for consistency)
        n_points = 200
        x_fixed = torch.rand(n_points, 1, dtype=torch.float32)
        x_fixed, _ = torch.sort(x_fixed)

        # PHASE 1: ADAM OPTIMIZATION (2000 epochs)
        print("\n" + "="*50)
        print("PHASE 1: Adam Optimization (2000 epochs)")
        print("="*50)

        for epoch in range(2000):
            # Always use Adam in Phase 1
            optimizer_name = "Adam"
            x_collocation = torch.rand(n_points, 1, dtype=torch.float32)

            # Store optimizer type
            self.optimizer_used.append(optimizer_name)

            # Compute losses
            L_momentum = self.momentum_loss(x_collocation)
            L_energy = self.energy_loss(x_collocation)
            L_mass = self.mass_transport_loss(x_collocation)
            L_boundary = self.boundary_loss()

            total_loss = (L_momentum + L_energy + L_mass + 100 * L_boundary)

            # Adam optimization step
            self.adam_optimizer.zero_grad()
            total_loss.backward()
            self.adam_optimizer.step()

            # Print Adam progress every 500 epochs
            if epoch % 500 == 0:
                print(f"Epoch {epoch:6d} | Adam  | Loss: {total_loss.item():.2e}")

            # Store history
            losses_history['total'].append(total_loss.item())
            losses_history['momentum'].append(L_momentum.item())
            losses_history['energy'].append(L_energy.item())
            losses_history['mass'].append(L_mass.item())
            losses_history['boundary'].append(L_boundary.item())
            losses_history['optimizer'].append(optimizer_name)

        # Record loss at the end of Adam phase
        adam_final_loss = losses_history['total'][-1]
        print(f"\nAdam phase complete. Final loss: {adam_final_loss:.4e}")

        # PHASE 2: LBFGS OPTIMIZATION (200 epochs)
        print("\n" + "="*50)
        print("PHASE 2: LBFGS Optimization (200 epochs)")
        print("="*50)

        self.current_collocation_points = x_fixed.clone()

        for epoch in range(200):
            # Always use LBFGS in Phase 2
            optimizer_name = "LBFGS"

            # Store optimizer type
            self.optimizer_used.append(optimizer_name)

            # LBFGS optimization step
            self.lbfgs_epoch_counter += 1

            # Run LBFGS step
            self.lbfgs_optimizer.step(self.lbfgs_closure)

            # Recompute loss after LBFGS
            # Original issue: Calling momentum_loss (and others) inside torch.no_grad()
            # caused RuntimeError as grad() needs a computation graph.
            # Removing the no_grad() block allows derivative calculation for re-evaluation.
            L_momentum = self.momentum_loss(self.current_collocation_points)
            L_energy = self.energy_loss(self.current_collocation_points)
            L_mass = self.mass_transport_loss(self.current_collocation_points)
            L_boundary = self.boundary_loss()
            total_loss = (L_momentum + L_energy + L_mass + 100 * L_boundary)
            current_loss = total_loss.item()

            # Print LBFGS progress every 20 epochs
            if epoch % 20 == 0:
                print(f"Epoch {2000 + epoch:6d} | LBFGS | Loss: {current_loss:.2e}")

            # Store history
            losses_history['total'].append(current_loss)
            losses_history['momentum'].append(L_momentum.item())
            losses_history['energy'].append(L_energy.item())
            losses_history['mass'].append(L_mass.item())
            losses_history['boundary'].append(L_boundary.item())
            losses_history['optimizer'].append(optimizer_name)

        # Final results and statistics
        print(f"\n{'='*75}")
        print("TRAINING COMPLETE - FIXED EPOCH OPTIMIZATION:")
        print(f"{'='*75}")

        adam_count = sum(1 for opt in self.optimizer_used if opt == 'Adam')
        lbfgs_count = sum(1 for opt in self.optimizer_used if opt == 'LBFGS')
        total_epochs_used = len(losses_history['total'])

        print(f"✓ Adam epochs: {adam_count} (Phase 1: 0-1999)")
        print(f"✓ LBFGS epochs: {lbfgs_count} (Phase 2: 2000-2199)")
        print(f"✓ Total epochs: {total_epochs_used}")
        print(f"✓ Loss after Adam: {adam_final_loss:.4e}")
        print(f"✓ Final Loss after LBFGS: {losses_history['total'][-1]:.4e}")

        # Calculate improvement
        improvement = (adam_final_loss - losses_history['total'][-1]) / adam_final_loss * 100
        print(f"✓ Improvement from LBFGS: {improvement:.1f}%")

        # Print loss reduction over LBFGS phase
        lbfgs_start_loss = losses_history['total'][1999]
        lbfgs_end_loss = losses_history['total'][-1]
        print(f"\nLBFGS Phase Details:")
        print(f"  - Starting loss: {lbfgs_start_loss:.4e}")
        print(f"  - Ending loss: {lbfgs_end_loss:.4e}")
        print(f"  - Reduction: {(lbfgs_start_loss - lbfgs_end_loss)/lbfgs_start_loss*100:.1f}%")
        print(f"  - Average loss per LBFGS epoch: {np.mean(losses_history['total'][2000:]):.4e}")

        return losses_history

    # =============================================================================
    # PART 9: SOLUTION COMPUTATION
    # =============================================================================
    def compute_solutions(self, n_points=300):
        """Compute complete solutions with derivatives"""
        x = torch.linspace(0, 1, n_points).reshape(-1, 1)

        x.requires_grad_(True)
        F, beta, phi = self.model(x)

        F_x = grad(F, x, torch.ones_like(F), create_graph=True)[0]
        F_xx = grad(F_x, x, torch.ones_like(F_x), create_graph=True)[0]
        beta_x = grad(beta, x, torch.ones_like(beta), create_graph=True)[0]
        phi_x = grad(phi, x, torch.ones_like(phi), create_graph=True)[0]

        return {
            'x': x.detach().numpy(),
            'F': F.detach().numpy(), 'F_x': F_x.detach().numpy(), 'F_xx': F_xx.detach().numpy(),
            'beta': beta.detach().numpy(), 'beta_x': beta_x.detach().numpy(),
            'phi': phi.detach().numpy(), 'phi_x': phi_x.detach().numpy()
        }

# =============================================================================
# PART 10: SEPARATE TOTAL LOSS PLOT WITH PHASE INDICATION
# =============================================================================
def plot_total_loss_separately(losses_history):
    """Create separate plot for total loss vs epochs with phase indication"""
    plt.figure(figsize=(14, 7))

    epochs = range(len(losses_history['total']))
    total_loss = losses_history['total']

    # Separate phases for plotting
    adam_epochs = list(range(2000))  # First 2000 epochs
    lbfgs_epochs = list(range(2000, 2200))  # Next 200 epochs

    adam_losses = losses_history['total'][:2000]
    lbfgs_losses = losses_history['total'][2000:]

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Full training curve
    ax1.semilogy(adam_epochs, adam_losses, 'b-', linewidth=2, label='Adam Phase', alpha=0.7)
    ax1.semilogy(lbfgs_epochs, lbfgs_losses, 'r-', linewidth=2, label='LBFGS Phase', alpha=0.7)
    ax1.axvline(x=1999.5, color='green', linestyle='--', linewidth=2,
               label=f'Phase Transition (Epoch 2000)')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Total Loss (log scale)', fontsize=12)
    ax1.set_title('Full Training: Adam (2000 epochs) → LBFGS (200 epochs)',
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')

    # Add text box with info
    info_text = f"Total Epochs: {len(losses_history['total'])}\n"
    info_text += f"Adam Epochs: {len(adam_losses)}\n"
    info_text += f"LBFGS Epochs: {len(lbfgs_losses)}\n"
    info_text += f"Final Loss: {total_loss[-1]:.4e}"

    ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Zoom on LBFGS phase
    ax2.semilogy(range(200), lbfgs_losses, 'r-', linewidth=3, label='LBFGS Phase')
    ax2.set_xlabel('LBFGS Epochs (0-199)', fontsize=12)
    ax2.set_ylabel('Total Loss (log scale)', fontsize=12)
    ax2.set_title('LBFGS Phase Detail (Epochs 2000-2199)',
                 fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')

    # Add markers for every 20th epoch
    for i in range(0, 200, 20):
        ax2.plot(i, lbfgs_losses[i], 'ko', markersize=4)
        if i % 40 == 0:
            ax2.annotate(f'Epoch {2000+i}\n{losses_history["total"][2000+i]:.1e}',
                        xy=(i, lbfgs_losses[i]),
                        xytext=(10, 20), textcoords='offset points',
                        fontsize=8, ha='center')

    plt.tight_layout()
    plt.savefig('fixed_epoch_optimization.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print optimization strategy summary
    print("\n" + "="*70)
    print("FIXED EPOCH OPTIMIZATION STRATEGY:")
    print("="*70)
    print("Phase 1: Adam Optimizer")
    print(f"  - Epochs: 0-1999 (2000 total)")
    print(f"  - Learning rate: 0.001")
    print(f"  - Final loss: {adam_losses[-1]:.4e}")

    print("\nPhase 2: LBFGS Optimizer")
    print(f"  - Epochs: 2000-2199 (200 total)")
    print(f"  - Learning rate: 0.1")
    print(f"  - Final loss: {lbfgs_losses[-1]:.4e}")
    print(f"  - Improvement: {((adam_losses[-1] - lbfgs_losses[-1])/adam_losses[-1]*100):.1f}%")

    print(f"\nTotal Training: {len(total_loss)} epochs")
    print(f"Final Model Loss: {total_loss[-1]:.4e}")
    print("="*70)

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    print("JEFFERY-HAMEL FLOW SOLVER")
    print("FIXED EPOCH OPTIMIZATION: 2000 Adam → 200 LBFGS")
    print("NO EARLY STOPPING - Fixed schedule")
    print("="*70)

    # Create solver
    solver = CompleteJefferyHamelSolver()

    # Train with fixed epoch optimization
    losses_history = solver.train()

    # Compute solutions
    solutions = solver.compute_solutions()

    # Plot total loss with phase indication
    plot_total_loss_separately(losses_history)

if __name__ == "__main__":
    main()
