from AutoEncoderObjects import EntropyLimitedAutoencoder, MSSSIMLoss, NLPDLoss, AutoEncodeData
from torch.nn import MSELoss

import pickle
import torch

from LoadingDefault import LoadAudiosTest
import pandas as pd

from preprocessing import adapt

with open("MODELOS/" + "metadatos_modelos.pkl", "rb") as f:
    lista_cargada = pickle.load(f)
    
mse_songs = EntropyLimitedAutoencoder()
nlpd_songs = EntropyLimitedAutoencoder()
msssim_songs = EntropyLimitedAutoencoder()

mse_noise = EntropyLimitedAutoencoder()
nlpd_noise = EntropyLimitedAutoencoder()
msssim_noise = EntropyLimitedAutoencoder()

mse_songs.load_state_dict(torch.load("MODELOS/MSE-songs.pth", map_location=torch.device('cpu')))
nlpd_songs.load_state_dict(torch.load("MODELOS/NLPD-songs.pth", map_location=torch.device('cpu')))
msssim_songs.load_state_dict(torch.load("MODELOS/MSSSIM-songs.pth", map_location=torch.device('cpu')))

mse_noise.load_state_dict(torch.load("MODELOS/MSE-noise.pth", map_location=torch.device('cpu')))
nlpd_noise.load_state_dict(torch.load("MODELOS/NLPD-noise.pth", map_location=torch.device('cpu')))
msssim_noise.load_state_dict(torch.load("MODELOS/MSSSIM-noise.pth", map_location=torch.device('cpu')))

XX, metadata = LoadAudiosTest(limit=None)
XX = adapt(XX)

##################################
criterions = (MSELoss, NLPDLoss, MSSSIMLoss)
numer_error = []
with torch.no_grad():
    for modelized in (mse_songs, nlpd_songs, msssim_songs, mse_noise, nlpd_noise, msssim_noise):
        modelized.eval(); modelized.to("cuda")
        dist_criterio = [[] for _ in range(len(criterions))]
        for element in range(len(XX)):
            XXX = XX[element:element+1]#.unsqueeze(0)
            resulted = AutoEncodeData(modelized, XXX.to("cuda"))
            for loss_f in range(len(criterions)):
                criterion = criterions[loss_f]()
                dist_criterio[loss_f].append(float(criterion(resulted.to("cuda"), XXX.to("cuda"))))
        
        for crit in dist_criterio:
            numer_error.append(sum(crit) / len(crit))
##################################


##################################
# with torch.no_grad():
#     numer_error = []
#     for modelized in (mse_songs, nlpd_songs, msssim_songs, mse_noise, nlpd_noise, msssim_noise):
#         modelized.eval(); modelized.to("cuda")
#         resulted = AutoEncodeData(modelized, XX.to("cuda"))
#         for loss_f in (MSELoss, NLPDLoss, MSSSIMLoss):
#             criterion = loss_f()
#             numer_error.append(float(criterion(resulted.to("cuda"), XX.to("cuda"))))
#         modelized.to("cpu")
##################################

list_song_noise = ["SONG"] * 9 + ["NOISE"] * 9
list_mse_nlpd_msssim = (["MSE"] * 3 + ["NLPD"] * 3 + ["MSSSIM"] * 3)*2
list_metric_test = ["MSE","NLPD","MSSSIM"] * 6


df = pd.DataFrame({
    "Training Data": list_song_noise,
    "Loss": list_mse_nlpd_msssim,
    "Test Metric": list_metric_test,
    "Value error": numer_error
})

print("Si se ejecuta esto es que se han cargado todos los datos<3")
print(df)

df = df.pivot(index=("Training Data",'Loss'), columns='Test Metric', values='Value error')
df = df[['MSE','NLPD',"MSSSIM"]]

df = df.sort_values(
    by=["Training Data", "Loss"], 
    ascending=[False, True],  
    key=lambda x: x if x.name != "Loss" else pd.Categorical(x, categories=["MSE", "NLPD"], ordered=True)
)

print(df)

with open("losses_table.pkl", "wb") as f:
    pickle.dump(df, f)