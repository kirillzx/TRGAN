import numpy as np
import torch
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from TRGAN.encoders import *
from TRGAN.TRGAN_main import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_cat_emb(X_oh, dim_Xoh, lr_E_oh, epochs=20, batch_size=2**8, load=False,\
                   directory='Pretrained_model/', names=['TRGAN_E_oh.pkl', 'TRGAN_D_oh.pkl', 'X_oh_emb.npy']):
    if load:
        encoder_onehot = Encoder_onehot(len(X_oh.columns), dim_Xoh).to(device)
        decoder_onehot = Decoder_onehot(dim_Xoh, len(X_oh.columns)).to(device)

        encoder_onehot.load_state_dict(torch.load(directory + names[0]))
        decoder_onehot.load_state_dict(torch.load(directory + names[1]))

        encoder_onehot.eval()
        decoder_onehot.eval()

        X_oh_emb = np.load(directory + names[2])
        # X_oh_emb = encoder_onehot(torch.FloatTensor(X_oh).to(device)).detach().cpu().numpy()

    else:
        X_oh_emb, encoder_onehot, decoder_onehot = create_categorical_embeddings(X_oh, dim_Xoh, lr_E_oh, epochs, batch_size)
        
        torch.save(encoder_onehot.state_dict(), directory + names[0])
        torch.save(decoder_onehot.state_dict(), directory + names[1])

        np.save(directory + names[2], X_oh_emb)

        encoder_onehot.eval()
        decoder_onehot.eval()

    return X_oh_emb, encoder_onehot, decoder_onehot

def create_client_emb(dim_X_cl, data, client_info, dim_Xcl, lr_E_cl, epochs=20, batch_size=2**8,\
            load=False, directory='Pretrained_model/', names=['TRGAN_E_cl.pkl', 'TRGAN_D_cl.pkl', 'X_cl.npy', 'scaler.joblib', 'label_enc.joblib']):
    
    if load:
        encoder_cl_emb = Encoder_client_emb(len(client_info), dim_X_cl).to(device)
        decoder_cl_emb = Decoder_client_emb(dim_X_cl, len(client_info)).to(device)

        encoder_cl_emb.load_state_dict(torch.load(directory + names[0]))
        decoder_cl_emb.load_state_dict(torch.load(directory + names[1]))

        encoder_cl_emb.eval()
        decoder_cl_emb.eval()

        X_cl = np.load(directory+names[2])
        # X_cl = encoder_cl_emb(torch.FloatTensor(client_info_for_emb).to(device)).detach().cpu().numpy()
        scaler_cl_emb = joblib.load(directory + names[3])
        label_encoders = joblib.load(directory + names[4])

    else:
        X_cl, encoder_cl_emb, decoder_cl_emb, scaler_cl_emb, label_encoders = create_client_embeddings(data, client_info, dim_Xcl, lr_E_cl, epochs, batch_size)
        
        torch.save(encoder_cl_emb.state_dict(), directory + names[0])
        torch.save(decoder_cl_emb.state_dict(), directory + names[1])

        np.save(directory + names[2], X_cl)
        joblib.dump(scaler_cl_emb, directory + names[3])
        joblib.dump(label_encoders, directory + names[4])

        encoder_cl_emb.eval()
        decoder_cl_emb.eval()

    return X_cl, encoder_cl_emb, decoder_cl_emb, scaler_cl_emb, label_encoders

def create_conditional_vector(data, X_emb, date_feature, time, dim_Vc_h, dim_bce, \
            name_client_id='customer', name_agg_feature='amount', lr_E_Vc=1e-3, epochs=15, batch_size=2**8, model_time='noise', n_splits=2, load=False,\
            directory='Pretrained_model/', names=['TRGAN_E_Vc.pkl', 'Vc.npy', 'BCE.npy'], opt_time=True, xi_array=[], q_array=[]):
    
    if load:
        encoder = Encoder(len(X_emb[0]), dim_Vc_h).to(device)
        encoder.load_state_dict(torch.load(directory + names[0]))
        encoder.eval()

        # cond_vector = np.load(directory + names[1])
        # synth_time = np.load(directory + names[2])
        # synth_time = pd.DataFrame(synth_time, columns=date_feature)
        # date_transformations = np.load(directory + names[3])

        X_emb_V_c = encoder(torch.FloatTensor(X_emb).to(device)).detach().cpu().numpy()

        behaviour_cl_enc = np.load(directory + names[2])

        cond_vector, synth_time, date_transformations, deltas_by_clients, synth_deltas_by_clients, xiP_array, idx_array = create_cond_vector_with_time_gen(X_emb_V_c, data, behaviour_cl_enc, date_feature, name_client_id, time,
                        model_time, n_splits, opt_time, xi_array, q_array)

    else:
        cond_vector, synth_time, date_transformations, behaviour_cl_enc, encoder, deltas_by_clients, synth_deltas_by_clients, xiP_array, idx_array = create_cond_vector(data, X_emb, date_feature, time, dim_Vc_h, dim_bce, \
                    name_client_id, name_agg_feature, lr_E_Vc, epochs, batch_size, model_time, n_splits, opt_time, xi_array, q_array)
        
        torch.save(encoder.state_dict(), directory + names[0])

        # np.save(directory + names[1], cond_vector)
        # np.save(directory + names[2], synth_time)
        # np.save(directory + names[3], date_transformations)
        np.save(directory + names[2], behaviour_cl_enc)

        encoder.eval()

    return cond_vector, synth_time, date_transformations, behaviour_cl_enc, encoder, deltas_by_clients, synth_deltas_by_clients, xiP_array, idx_array 


def create_cont_emb(dim_X_cont, data, cont_features, lr_E_cont=1e-3, epochs=20, batch_size=2**8,\
            load=False, directory='Pretrained_model/', names='scaler_cont', type_scale='Autoencoder'):
    
    if load:
        scaler_cont = list(np.load(directory + names, allow_pickle=True))

        encoder_cont_emb = Encoder_cont_emb(len(cont_features), dim_X_cont).to(device)

        encoder_cont_emb.load_state_dict(scaler_cont[-1].state_dict())
        encoder_cont_emb.eval()

        scaler_cont[2].reset_randomization()
        X_cont = scaler_cont[2].transform(data[cont_features])
        X_cont = scaler_cont[1].transform(X_cont)
  
        X_cont = encoder_cont_emb(torch.FloatTensor(X_cont).to(device)).detach().cpu().numpy()

    else:
        X_cont, scaler_cont = preprocessing_cont(data, cont_features, type_scale=type_scale, lr=lr_E_cont, bs=batch_size, epochs=epochs, dim_cont_emb=dim_X_cont)
        
        # torch.save(scaler_cont[-1].state_dict(), directory + names[0])
        # torch.save(scaler_cont[0].state_dict(), directory + names[1])

        # # np.save(directory + names[2], X_cl)
        # joblib.dump(scaler_cont[1], directory + names[3])
        # joblib.dump(scaler_cont[2], directory + names[4])

        np.save(directory + names, scaler_cont)

    return X_cont, scaler_cont