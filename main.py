from src.train.train import *
from pathlib import Path

env = gym.make('CartPole-v1', render_mode='human')
checkpoint_path = Path.cwd() / 'checkpoint' / 'checkpoint.tar'
checkpoint = torch.load(checkpoint_path)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = QNetwork()
model.load_state_dict(checkpoint['model'])
model.to(device)

done = False
s, _ = env.reset()

while not done:
    with torch.no_grad():
        q_value = model(torch.tensor(s, dtype=torch.float32, device=device))
        a = q_value.argmax().item()

    s, _, d1, d2, _ = env.step(a)
    done = d1 or d2

env.close()
