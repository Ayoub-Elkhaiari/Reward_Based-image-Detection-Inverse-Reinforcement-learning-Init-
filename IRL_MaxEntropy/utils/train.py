import torch
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss

def irl_loss(expert_rewards, sampled_rewards):
    """Maximum entropy IRL loss function."""
    logits = torch.cat([expert_rewards, sampled_rewards], dim=0)
    labels = torch.cat([torch.ones_like(expert_rewards), torch.zeros_like(sampled_rewards)], dim=0)
    return BCEWithLogitsLoss()(logits, labels)

def train_irl(reward_net, images, expert_bboxes, epochs=5, lr=1e-3):
    optimizer = optim.Adam(reward_net.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0
        for img, expert_bbox in zip(images, expert_bboxes):
            img_tensor = torch.tensor(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            expert_bbox_tensor = torch.tensor(expert_bbox).float().unsqueeze(0)
            random_bbox = torch.rand(1, 4) * 64  
            expert_reward = reward_net(img_tensor, expert_bbox_tensor)
            sampled_reward = reward_net(img_tensor, random_bbox)
            loss = irl_loss(expert_reward, sampled_reward)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(images):.4f}")

def train_detector(detector, reward_net, images, epochs=100, lr=1e-3):
    optimizer = optim.Adam(detector.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0
        for img in images:
            img_tensor = torch.tensor(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            pred_bbox = detector(img_tensor)
            reward = reward_net(img_tensor, pred_bbox)
            loss = -reward.mean()  
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(images):.4f}")
