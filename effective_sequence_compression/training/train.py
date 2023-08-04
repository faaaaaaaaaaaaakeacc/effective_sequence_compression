import torch
from tqdm.auto import trange

def train(model, dataset, epochs, logger, criterion, metric, device):
    optimizer = torch.optim.Adam(model.parameters(), lr = 3e-4)
    for _ in trange(epochs):
        for batch in dataset.iter_batches_train():
            optimizer.zero_grad()
            output = model(batch.text)
            loss = criterion(output, torch.tensor(batch.target).to(device))
            loss.backward()
            logger.log_loss(loss.item())
            optimizer.step()
        with torch.no_grad():
            for batch in dataset.iter_batches_test():
                output = model(batch.text)
                metric(output.cpu().detach().numpy(), batch.target)
            logger.log_metric(metric.finish())
