from LoadingDefault import LoadData, LoadNoise
from AutoEncoderObjects import EntropyLimitedAutoencoder, MSSSIMLoss, NLPDLoss
from torch.nn import MSELoss
import torch.optim as optim
import torch
import pickle

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

num_epochs = 5

dataloader = LoadData()
noise = LoadNoise()

metadata_model = []

loss_functions = (MSELoss, NLPDLoss, MSSSIMLoss)

for data in (dataloader, noise):
    for loss_f in range(len(loss_functions)):

        ae = EntropyLimitedAutoencoder()
        criterion = loss_functions[loss_f]()
        optimizer = optim.AdamW(ae.parameters(), lr=1e-3, weight_decay=1e-4)

        mod = {}
        if loss_f == 0:
            mod["loss_type"] = "MSE"
        elif loss_f == 1:
            mod["loss_type"] = "NLPD"
        elif loss_f == 2:
            mod["loss_type"] = "MSSSIM"
        
        if data == dataloader:
            mod["data"] = "songs"
        else:
            mod["data"] = "noise"

        mod["epochs"] = num_epochs

        loss_epochs = []
        loss_batch = []

        for epoch in range(num_epochs):
            total_loss = 0
            for batch in data:  # dataloader ya tiene los batches de 64x1x256x256
                batch = batch[0] # Extraer tensor

                optimizer.zero_grad()  # Reiniciar gradientes

                outputs = ae(batch)  # Forward pass
                loss = criterion(outputs, batch)  # Comparar con entrada

                loss.backward()  # Backpropagation
                optimizer.step()  # Actualizar pesos

                total_loss += loss.item()
                loss_batch.append(loss.item())

            avg_loss = total_loss / len(data)
            loss_epochs.append(avg_loss)

            print(f"Época [{epoch+1}/{num_epochs}], Pérdida: {avg_loss:.6f}")
        
        mod["loss_epochs"] = loss_epochs
        mod["loss_batch"] = loss_batch
        mod["file_weights"] = mod["loss_type"] + "-" + mod["data"] + ".pth"
        mod["file_full"] = mod["loss_type"] + "-" + mod["data"] + "-completo.pth"

        torch.save(ae.state_dict(), "MODELOS/" + mod["file_weights"])
        torch.save(ae, "MODELOS/" + mod["file_full"])
    
        metadata_model.append(mod)

with open("MODELOS/" + "metadatos_modelos.pkl", "wb") as f:
    pickle.dump(metadata_model, f)