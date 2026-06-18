import torch.optim as optim

def train_manifold_estimator(data_loader, d_dim, p_dim, num_charts=5, epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    generator = GenerativeManifoldEstimator(num_charts, d_dim, p_dim).to(device)
    discriminator = Discriminator(p_dim).to(device)
    
    # Optimizers
    opt_G = optim.Adam(generator.parameters(), lr=1e-4)
    opt_D = optim.Adam(discriminator.parameters(), lr=1e-4)
    
    lambda_consistency = 10.0 # Weight for C_N(g, \varphi)
    
    for epoch in range(epochs):
        for real_data in data_loader:
            real_data = real_data.to(device)
            batch_size = real_data.size(0)
            
            # ==========================================
            # 1. Train Discriminator (Maximize WGAN Loss)
            # ==========================================
            opt_D.zero_grad()
            
            fake_data = generator.generate(batch_size).detach()
            
            # WGAN Loss: E[D(real)] - E[D(fake)]
            # (Note: Requires Gradient Penalty or Weight Clipping in practice to enforce Lipschitz bound)
            d_loss = -(torch.mean(discriminator(real_data)) - torch.mean(discriminator(fake_data)))
            
            d_loss.backward()
            opt_D.step()
            
            # ==========================================
            # 2. Train Generators, Inverses, and Alphas
            # ==========================================
            opt_G.zero_grad()
            
            # A. Adversarial Loss
            fake_data = generator.generate(batch_size)
            g_adv_loss = -torch.mean(discriminator(fake_data))
            
            # B. Structural Consistency Loss C_N (Inverse Mapping Penalty)
            # sum_i || \varphi_i(g_i(z)) - z ||^2
            consistency_loss = 0
            z = generator.sample_latent(batch_size).to(device)
            for chart in generator.charts:
                x_projected = chart.g(z)
                z_reconstructed = chart.phi(x_projected)
                consistency_loss += F.mse_loss(z_reconstructed, z)
                
            # C. Spectral Regularization R(g)
            # (In a full implementation, you would compute the eigenvalues of the 
            # Jacobian here to ensure volume is preserved. Omitted for brevity.)
            
            g_total_loss = g_adv_loss + (lambda_consistency * consistency_loss)
            
            g_total_loss.backward()
            opt_G.step()
            
        print(f"Epoch {epoch} | D Loss: {d_loss.item():.4f} | G Loss: {g_total_loss.item():.4f}")

    return generator