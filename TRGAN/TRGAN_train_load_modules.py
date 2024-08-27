import numpy as np
import torch
import joblib
from TRGAN.encoders import *
from TRGAN.TRGAN_main_V2 import *


def embeddings(data: pd.DataFrame, cat_feat_names, num_feat_names, onehot_cols, date_feature, client_id,
               latent_dim, device: str = 'cpu', epochs: int = 80, load: bool = True, directory: str = 'Pretrained_model/'):
    
    if load:
        # onehot embeddings
        ######################################
        X_oh = pd.DataFrame(np.load(directory + 'X_oh.npy'), columns=np.load(directory + 'X_oh_cols.npy', allow_pickle=True))
        onehot_emb = np.load(directory + 'onehot_emb.npy')
        
        encoder_onehot = Encoder_onehot(len(X_oh.columns), latent_dim['onehot']).to(device)
        decoder_onehot = Decoder_onehot(latent_dim['onehot'], len(X_oh.columns)).to(device)
        
        encoder_onehot.load_state_dict(torch.load(directory + 'onehot_encoder.pt'))
        decoder_onehot.load_state_dict(torch.load(directory + 'onehot_decoder.pt'))
        
        encoder_onehot.eval()
        decoder_onehot.eval()
        
        scaler_onehot = {'encoder': encoder_onehot,
                         'decoder': decoder_onehot}
        ######################################
        
        
        
        # categorical embeddings
        ######################################
        categorical_emb = np.load(directory + 'categorical_emb.npy')
        
        encoder_cat = Encoder_client_emb(len(cat_feat_names), latent_dim['categorical']).to(device)
        decoder_cat = Decoder_client_emb(latent_dim['categorical'], len(cat_feat_names)).to(device)
        
        encoder_cat.load_state_dict(torch.load(directory + 'categorical_encoder.pt'))
        decoder_cat.load_state_dict(torch.load(directory + 'categorical_decoder.pt'))
        
        encoder_cat.eval()
        decoder_cat.eval()
        
        
        scaler_cat = {'encoder': encoder_cat,
                      'decoder': decoder_cat,
                      'scaler': joblib.load(directory + 'scaler_cat.joblib'),
                      'freq_encoder': joblib.load(directory + 'freq_encoder.joblib')}
        ######################################
        
        
        # numerical embeddings
        ######################################
        numerical_emb = np.load(directory + 'numerical_emb.npy')
        
        encoder_num = Encoder_cont_emb(len(num_feat_names), latent_dim['numerical']).to(device)
        decoder_num = Decoder_cont_emb(latent_dim['numerical'], len(num_feat_names)).to(device)
        
        encoder_num.load_state_dict(torch.load(directory + 'numerical_encoder.pt'))
        decoder_num.load_state_dict(torch.load(directory + 'numerical_decoder.pt'))
        
        encoder_num.eval()
        decoder_num.eval()
        
        scaler_num = {'encoder': encoder_num,
                      'decoder': decoder_num,
                      'scaler_minmax': joblib.load(directory + 'scaler_minmax.joblib'),
                      'scaler': joblib.load(directory + 'scaler_num.joblib')}
        ######################################
        
        
        # embeddings
        ######################################
        X_emb = np.load(directory + 'X_emb.npy')
        scaler = joblib.load(directory + 'scaler_emb.joblib')
        ######################################
        
        
        
        # create conditional vector and synth date
        ######################################
        encoder_cv = Encoder(len(X_emb[0]), latent_dim['cv']).to(device)
        encoder_cv.load_state_dict(torch.load(directory + 'cv_encoder.pt'))
        encoder_cv.eval()
        
        cv_params = {'deltas_real': np.load(directory + 'deltas_real.npy', allow_pickle=True),
                     'deltas_synth': np.load(directory + 'deltas_synth.npy', allow_pickle=True),
                     'xiP': np.load(directory + 'xiP.npy', allow_pickle=True),
                     'quantile_index': np.load(directory + 'quantile_index.npy', allow_pickle=True),
                     'date_transform': np.load(directory + 'date_transform.npy'),
                     'encoder': encoder_cv}
        
        cond_vector = encoder_cv(torch.FloatTensor(X_emb).to(device)).detach().cpu().numpy()
        cond_vector = np.concatenate([cond_vector, cv_params['date_transform']], axis=1)
        synth_date = pd.DataFrame(np.load(directory + 'synth_date.npy', allow_pickle=True), columns=[date_feature])
        ######################################
        
        
        
        
    else:
        # onehot embeddings
        ######################################
        X_oh = create_onehot(data, onehot_cols)
        onehot_emb, scaler_onehot = encode_onehot_embeddings(X_oh, latent_dim['onehot'], device=device, epochs=epochs)
        
        torch.save(scaler_onehot['encoder'].state_dict(), directory + 'onehot_encoder.pt')
        torch.save(scaler_onehot['decoder'].state_dict(), directory + 'onehot_decoder.pt')
        np.save(directory + 'onehot_emb.npy', onehot_emb)
        np.save(directory + 'X_oh.npy', X_oh)
        np.save(directory + 'X_oh_cols.npy', X_oh.columns.values)
        ######################################
        
        
        # categorical embeddings
        ######################################
        categorical_emb, scaler_cat = encode_categorical_embeddings(data, cat_feat_names, latent_dim=latent_dim['categorical'], device=device, epochs=epochs)
        
        torch.save(scaler_cat['encoder'].state_dict(), directory + 'categorical_encoder.pt')
        torch.save(scaler_cat['decoder'].state_dict(), directory + 'categorical_decoder.pt')
        np.save(directory + 'categorical_emb.npy', categorical_emb)
        joblib.dump(scaler_cat['scaler'], directory + 'scaler_cat.joblib')
        joblib.dump(scaler_cat['freq_encoder'], directory + 'freq_encoder.joblib')
        ######################################
        
        
        # numerical embeddings
        ######################################
        numerical_emb, scaler_num = encode_continuous_embeddings(data, num_feat_names, latent_dim=latent_dim['numerical'], device=device, epochs=epochs)
        
        torch.save(scaler_num['encoder'].state_dict(), directory + 'numerical_encoder.pt')
        torch.save(scaler_num['decoder'].state_dict(), directory + 'numerical_decoder.pt')
        np.save(directory + 'numerical_emb.npy', numerical_emb)
        joblib.dump(scaler_num['scaler_minmax'], directory + 'scaler_minmax.joblib')
        joblib.dump(scaler_num['scaler'], directory + 'scaler_num.joblib')
        ######################################
            
            
        # join embeddings
        ######################################
        X_emb = create_embeddings(onehot_emb, categorical_emb, numerical_emb)
        scaler = MinMaxScaler((-1, 1))
        X_emb = scaler.fit_transform(X_emb)
        
        np.save(directory + 'X_emb.npy', X_emb)
        joblib.dump(scaler, directory + 'scaler_emb.joblib')
        ######################################
        
        
        # create conditional vector and synth date
        ######################################
        print('Optimizing poisson intensity...')
        cond_vector, synth_date, cv_params = \
            create_cond_vector(data, X_emb, date_feature, client_id, time_type='synth', latent_dim=latent_dim['cv'], opt_time=True, device=device)
            
        torch.save(cv_params['encoder'].state_dict(), directory + 'cv_encoder.pt')
        np.save(directory + 'deltas_real.npy', cv_params['deltas_real'])
        np.save(directory + 'deltas_synth.npy', cv_params['deltas_synth'])
        np.save(directory + 'xiP.npy', cv_params['xiP'])
        np.save(directory + 'quantile_index.npy', cv_params['quantile_index'])   
        np.save(directory + 'date_transform.npy', cv_params['date_transform']) 
        np.save(directory + 'synth_date.npy', synth_date.values)  
        ######################################
        
    return X_emb, X_oh, cond_vector, synth_date, scaler_cat, scaler_onehot, scaler_num, cv_params, scaler
    
    
    
    
def train(X_emb, cond_vector, latent_dim, dim_noise=15, epochs=40, experiment_id='TRGAN_V2_1',
          DIRECTORY='Pretrained_model/', DEVICE='cpu', load=False):
    
    dim_X_emb = latent_dim['onehot'] + latent_dim['categorical'] + latent_dim['numerical']
    dim_Vc = latent_dim['cv'] + 5
    h_dim = 2**6
    num_blocks_gen = 2
    num_blocks_dis = 2

    if load:
        generator = Generator(dim_noise + dim_Vc, dim_X_emb, h_dim, num_blocks_gen).to(DEVICE)
        supervisor = Supervisor(dim_X_emb + dim_Vc, dim_X_emb, h_dim, num_blocks_gen).to(DEVICE)

        generator.load_state_dict(torch.load(f'{DIRECTORY}TRGAN_generator_exp_{experiment_id}.pt', map_location=DEVICE, weights_only=True))
        supervisor.load_state_dict(torch.load(f'{DIRECTORY}TRGAN_supervisor_exp_{experiment_id}.pt', map_location=DEVICE, weights_only=True))

        generator.eval()
        supervisor.eval()

        loss_array = np.load(f'{DIRECTORY}loss_array_exp_{experiment_id}.npy')

    else:
        generator, supervisor, loss_array, discriminator, discriminator2 = train_generator(X_emb, cond_vector, dim_Vc, dim_X_emb, dim_noise,\
                                        batch_size=2**9, lr_rates=[3e-4, 3e-4, 3e-4, 3e-4], num_epochs=epochs, num_blocks_gen=num_blocks_gen,\
                                        num_blocks_dis=num_blocks_dis, h_dim=h_dim, lambda1=3, alpha=0.7, device=DEVICE)
        
        torch.save(generator.state_dict(), f'{DIRECTORY}TRGAN_generator_exp_{experiment_id}.pt')
        torch.save(supervisor.state_dict(), f'{DIRECTORY}TRGAN_supervisor_exp_{experiment_id}.pt')

        np.save(f'{DIRECTORY}loss_array_exp_{experiment_id}.npy', loss_array)

        generator.eval()
        supervisor.eval()
        
    return generator, supervisor, loss_array 


# def sample(n_samples):
#     synth_data, synth_date, params = sample(n_samples, generator, supervisor, dim_noise, cond_vector, X_emb, cv_params['encoder'], data,\
#                                     date_feature, client_id, cv_params=cv_params, device=DEVICE)

#     synth_df = inverse_transform(synth_data, latent_dim, X_oh.columns, scaler_onehot, scaler_cat, scaler_num, cat_feat_names,
#                                 mcc_name, num_feat_names, True, synth_date, 'TRANS_TIME')







def create_cat_emb(X_oh, dim_Xoh, lr_E_oh, epochs=20, batch_size=2**8, load=False,\
                   directory='Pretrained_model/', names=['TRGAN_E_oh.pkl', 'TRGAN_D_oh.pkl', 'X_oh_emb.npy'], device='cpu', eps=2):
    if load:
        encoder_onehot = Encoder_onehot(len(X_oh.columns), dim_Xoh).to(device)
        decoder_onehot = Decoder_onehot(dim_Xoh, len(X_oh.columns)).to(device)

        encoder_onehot.load_state_dict(torch.load(directory + names[0]))
        decoder_onehot.load_state_dict(torch.load(directory + names[1]))

        encoder_onehot.eval()
        decoder_onehot.eval()

        # X_oh_emb = np.load(directory + names[2])
        X_oh_emb = encoder_onehot(torch.FloatTensor(X_oh.values).to(device)).detach().cpu().numpy()

    else:
        X_oh_emb, encoder_onehot, decoder_onehot = create_categorical_embeddings(X_oh, dim_Xoh, lr_E_oh, epochs, batch_size, device, eps)
        
        torch.save(encoder_onehot.state_dict(), directory + names[0])
        torch.save(decoder_onehot.state_dict(), directory + names[1])

        np.save(directory + names[2], X_oh_emb)

        encoder_onehot.eval()
        decoder_onehot.eval()

    return X_oh_emb, encoder_onehot, decoder_onehot

def create_client_emb(dim_X_cl, data1, client_info, dim_Xcl, lr_E_cl, epochs=20, batch_size=2**8,\
            load=False, directory='Pretrained_model/', names=['TRGAN_E_cl.pkl', 'TRGAN_D_cl.pkl', 'X_cl.npy', 'scaler.joblib', 'label_enc.joblib'],\
            device='cpu', eps=0.5):
    
    data = copy.deepcopy(data1)

    if load:
        encoder_cl_emb = Encoder_client_emb(len(client_info), dim_X_cl).to(device)
        decoder_cl_emb = Decoder_client_emb(dim_X_cl, len(client_info)).to(device)

        encoder_cl_emb.load_state_dict(torch.load(directory + names[0]))
        decoder_cl_emb.load_state_dict(torch.load(directory + names[1]))

        encoder_cl_emb.eval()
        decoder_cl_emb.eval()

        # X_cl = np.load(directory+names[2])

        scaler_cl_emb = joblib.load(directory + names[3])
        label_encoders = joblib.load(directory + names[4])
        client_info_new_features = []

        for i in range(len(client_info)):
            customer_enc = label_encoders[i].transform(data)[client_info[i]].values
            client_info_new_features.append(customer_enc)

        client_info_for_emb = np.array(client_info_new_features).T
        client_info_for_emb = client_info_for_emb.astype(float)
        client_info_for_emb = scaler_cl_emb.transform(client_info_for_emb)

        X_cl = encoder_cl_emb(torch.FloatTensor(client_info_for_emb).to(device)).detach().cpu().numpy()
        

    else:
        X_cl, encoder_cl_emb, decoder_cl_emb, scaler_cl_emb, label_encoders = create_client_embeddings(data, client_info, dim_Xcl,\
                                                                                            lr_E_cl, epochs, batch_size, device, eps)
        
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
            directory='Pretrained_model/', names=['TRGAN_E_Vc.pkl', 'Vc.npy', 'BCE.npy'], opt_time=True, xi_array=[], q_array=[], device='cpu', eps=0.5):
    
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
        cond_vector, synth_time, date_transformations, behaviour_cl_enc, encoder, deltas_by_clients,\
        synth_deltas_by_clients, xiP_array, idx_array =\
              create_cond_vector(data, X_emb, date_feature, time, dim_Vc_h, dim_bce, name_client_id, name_agg_feature,\
                                lr_E_Vc, epochs, batch_size, model_time, n_splits, opt_time, xi_array, q_array, device, eps)
        
        torch.save(encoder.state_dict(), directory + names[0])

        # np.save(directory + names[1], cond_vector)
        # np.save(directory + names[2], synth_time)
        # np.save(directory + names[3], date_transformations)
        np.save(directory + names[2], behaviour_cl_enc)

        encoder.eval()

    return cond_vector, synth_time, date_transformations, behaviour_cl_enc, encoder, deltas_by_clients, synth_deltas_by_clients, xiP_array, idx_array 


def create_cont_emb(dim_X_cont, data, cont_features, lr_E_cont=1e-3, epochs=20, batch_size=2**8,\
            load=False, directory='Pretrained_model/', names='scaler_cont', type_scale='Autoencoder', device='cpu', eps=0.5):
    
    if load:
        if type_scale == 'Autoencoder':
            scaler_cont = list(np.load(directory + names, allow_pickle=True))

            encoder_cont_emb = Encoder_cont_emb(len(cont_features), dim_X_cont).to(device)

            encoder_cont_emb.load_state_dict(scaler_cont[-1].state_dict())
            encoder_cont_emb.eval()

            scaler_cont[2].reset_randomization()
            X_cont = scaler_cont[2].transform(data[cont_features])
            X_cont = scaler_cont[1].transform(X_cont)
    
            X_cont = encoder_cont_emb(torch.FloatTensor(X_cont).to(device)).detach().cpu().numpy()
        
        elif type_scale == 'CBNormalize':
            scaler_cont = list(np.load(directory + names, allow_pickle=True))
            X_cont = scaler_cont[0].transform(data)['amount.normalized'].values.reshape(-1, 1)

        elif type_scale == 'Standardize':
            scaler_cont = list(np.load(directory + names, allow_pickle=True))
            X_cont = scaler_cont[0].transform(data[cont_features])


    else:
        X_cont, scaler_cont = preprocessing_cont(data, cont_features, type_scale=type_scale, lr=lr_E_cont,\
                                                bs=batch_size, epochs=epochs, dim_cont_emb=dim_X_cont, device=device, eps=eps)
        
        # torch.save(scaler_cont[-1].state_dict(), directory + names[0])
        # torch.save(scaler_cont[0].state_dict(), directory + names[1])

        # # np.save(directory + names[2], X_cl)
        # joblib.dump(scaler_cont[1], directory + names[3])
        # joblib.dump(scaler_cont[2], directory + names[4])

        np.save(directory + names, scaler_cont, allow_pickle=True)

    return X_cont, scaler_cont