import numpy as np
import torch
from astropy.table import Table

class APerr():
    """ 
    apogee dr16 aspcap instance
    """

    def __init__(self, npydata, total_num=6000, part_train=True, device=torch.device('cpu')):
        self.dic = np.load(npydata, allow_pickle=True)
        self.total_num = total_num
        self.part_train = part_train
        self.device = device
                
    def __len__(self) -> int:
        if self.part_train:
            num_sets = self.total_num
        else:
            num_sets = len(self.dic)
        return num_sets
    
    def __getitem__(self, idx: int):
        # idx = self.dic[idx]['OBJ']
        flux = self.dic[idx]['flux']
        e_flux = self.dic[idx]['fluxerr']
        wave = self.dic[idx]['wave']
        
        prlx, e_prlx = self.dic[idx]['Gaia_parallax'], self.dic[idx]['Gaia_parallax_err']
        prlx_hogg, e_prlx_hogg = self.dic[idx]['spec_parallax'], self.dic[idx]['spec_parallax_err']

        flux = np.where(flux>0., flux, 0.)
        flux    = torch.tensor(flux.reshape(-1,1).astype(np.float32))
        # prlx  = torch.tensor(prlx.reshape(-1,1).astype(np.float32))
        output = np.vstack([prlx, e_prlx])
        output = torch.tensor(output.reshape(-1,1).astype(np.float32))
        return flux.to(self.device), output.to(self.device)

class AP_norm_old():
    """ 
    apogee dr16 aspcap instance
    """
    def __init__(self, npydata, total_num=6000, device=torch.device('cpu')):
        self.dic = np.load(npydata, allow_pickle=True)
        self.total_num = total_num
        self.device = device
    
                
    def __len__(self) -> int:
        # num_sets = len(self.dic)
        num_sets = self.total_num
        return num_sets
    
    def __getitem__(self, idx: int):
        # idx = self.dic[idx]['OBJ']
        flux = self.dic[idx]['flux']
        e_flux = self.dic[idx]['fluxerr']
        wave = self.dic[idx]['wave']
        
        prlx, e_prlx = self.dic[idx]['Gaia_parallax'], self.dic[idx]['Gaia_parallax_err']
        prlx_hogg, e_prlx_hogg = self.dic[idx]['spec_parallax'], self.dic[idx]['spec_parallax_err']

        flux = np.where(flux>0., flux, 0.)
        flux    = torch.tensor(flux.reshape(-1,1).astype(np.float32))
        prlx  = torch.tensor(prlx.reshape(-1,1).astype(np.float32))
        return flux.to(self.device), prlx.to(self.device)
    
    
    
class AP_norm():
    """ 
    apogee dr14 apogee instance
    """
    def __init__(self, npydata, total_num=6000, part_train=True, device=torch.device('cpu')):
        self.dic = np.load(npydata, allow_pickle=True)
        self.total_num = total_num
        self.device = device
        self.part_train = part_train
    
                
    def __len__(self) -> int:
        if self.part_train:
            num_sets = self.total_num
        else:
            num_sets = len(self.dic)
        return num_sets
    
    def __getitem__(self, idx: int):
        # idx = self.dic[idx]['OBJ']
        flux = self.dic[idx]['norm_spec']
        e_flux = self.dic[idx]['norm_spec_err']
        
        prlx, e_prlx = self.dic[idx]['Gaia_parallax'], self.dic[idx]['Gaia_parallax_err']
        prlx_hogg, e_prlx_hogg = self.dic[idx]['spec_parallax'], self.dic[idx]['spec_parallax_err']

        mag = self.dic[idx]['mag']

        flux = np.where(flux>0., flux, 1e-3)
        inpt = np.hstack([np.log(flux), mag])
        inpt = torch.tensor(inpt.reshape(-1,1).astype(np.float32))
        # flux    = torch.tensor(flux.reshape(-1,1).astype(np.float32))
        # prlx  = torch.tensor(prlx.reshape(-1,1).astype(np.float32))
        output = np.vstack([float(prlx), float(e_prlx)])
        output = torch.tensor(output.reshape(-1,1).astype(np.float32))

        return {'x':inpt.to(self.device), 'y':output.to(self.device), 'id':self.dic[idx]['tmass_id']}


class AP_cat():
    """ 
    apogee dr14 apogee instance
    """
    def __init__(self, npydata, cat_name=None, total_num=6000, part_train=True,  device=torch.device('cpu')):
        self.dic = np.load(npydata, allow_pickle=True)
        self.ids = np.array([d['tmass_id'] for d in self.dic])
        self.total_num = total_num
        self.device = device
        self.part_train = part_train
        self.cat = Table.read(cat_name)
    
                
    def __len__(self) -> int:
        if self.part_train:
            num_sets = self.total_num
        else:
            num_sets = len(self.cat)
        return num_sets
    
    def __getitem__(self, idx: int):
        match = np.nonzero(self.ids==self.cat['2MASS_ID'][idx])[0][0]

        flux = self.dic[match]['norm_spec']
        # e_flux = self.dic[match]['norm_spec_err']
        prlx, e_prlx = self.dic[match]['Gaia_parallax'], self.dic[match]['Gaia_parallax_err']
        # prlx_hogg, e_prlx_hogg = self.dic[match]['spec_parallax'], self.dic[match]['spec_parallax_err']

        mag = self.dic[match]['mag']
        flux = np.where(flux>0., flux, 1e-3)
        inpt = np.hstack([np.log(flux), mag])
        inpt = torch.tensor(inpt.reshape(-1,1).astype(np.float32))
        output = np.vstack([float(prlx), float(e_prlx)])
        output = torch.tensor(output.reshape(-1,1).astype(np.float32))

        return {'x':inpt.to(self.device), 'y':output.to(self.device), 'id':self.dic[match]['tmass_id']}


class AP_norm_mag():
    """ 
    apogee dr14 apogee instance
    """
    def __init__(self, npydata, total_num=6000, part_train=True, device=torch.device('cpu')):
        self.dic = np.load(npydata, allow_pickle=True)
        self.total_num = total_num
        self.device = device
        self.part_train = part_train
    
                
    def __len__(self) -> int:
        if self.part_train:
            num_sets = self.total_num
        else:
            num_sets = len(self.dic)
        return num_sets
    
    def __getitem__(self, idx: int):
        # idx = self.dic[idx]['OBJ']
        flux = self.dic[idx]['norm_spec']
        e_flux = self.dic[idx]['norm_spec_err']
        
        prlx, e_prlx = self.dic[idx]['Gaia_parallax'], self.dic[idx]['Gaia_parallax_err']
        prlx_hogg, e_prlx_hogg = self.dic[idx]['spec_parallax'], self.dic[idx]['spec_parallax_err']

        mag = self.dic[idx]['mag']

        flux = np.where(flux>0., flux, 1e-3)
        inpt = np.hstack([np.log(flux), mag])
        inpt = torch.tensor(inpt.reshape(-1,1).astype(np.float32))

        # output = np.vstack([float(prlx), float(e_prlx)])
        absmag_pseudo = prlx * np.power(10, 0.2*mag)
        e_absmag_pseudo = e_prlx *  np.power(10, 0.2*mag)
        output = np.vstack([absmag_pseudo, e_absmag_pseudo])
        output = torch.tensor(output.reshape(-1,1).astype(np.float32))

        return inpt.to(self.device), output.to(self.device)


class AP_fakeprlx():
    """ 
    apogee dr16 aspcap instance
    """

    def __init__(self, npydata, total_num=6000, part_train=True, device=torch.device('cpu')):
        self.dic = np.load(npydata, allow_pickle=True)

        self.fake_prlx = torch.rand(len(self.dic), generator=torch.Generator().manual_seed(42))
        self.total_num = total_num
        self.part_train = part_train
        self.device = device
                
    def __len__(self) -> int:
        if self.part_train:
            num_sets = self.total_num
        else:
            num_sets = len(self.dic)
        return num_sets
    
    def __getitem__(self, idx: int):
        # idx = self.dic[idx]['OBJ']
        flux = self.dic[idx]['norm_spec']
        flux = np.where(flux>0., flux, 0.)
        flux    = torch.tensor(flux.reshape(-1,1).astype(np.float32))

        prlx = self.fake_prlx[idx]
        output = np.vstack([prlx, 0.1*prlx])
        output = torch.tensor(output.reshape(-1,1).astype(np.float32))
        return {'x':flux.to(self.device), 'y':output.to(self.device)}


class GaiaXPlabel():
    """Gaia DR3 XP spectrum to stellar labels instance
    """
    def __init__(self, npydata, total_num=6000, part_train=True, device=torch.device('cpu')) -> None:
        self.data = np.load(npydata, allow_pickle=True).item()
        self.spec = self.data['norm_spec']
        self.total_num = total_num
        self.part_train = part_train
        self.device = device
    
    def __len__(self) -> int:
        if self.part_train:
            num_sets = self.total_num
        else:
            num_sets = len(self.spec)
        return num_sets

    def __getitem__(self, idx: int):

        lnflux = self.spec[idx,:,1]
        lnflux[np.isnan(lnflux)] = np.mean(lnflux)
        lnflux    = torch.tensor(lnflux.reshape(-1,1).astype(np.float32))

        moh = self.data['moh'][idx]
        aom = self.data['aom'][idx]

        output = np.vstack([moh, aom])
        output = torch.tensor(output.reshape(-1,1).astype(np.float32))
        return {'x':lnflux.to(self.device), 'y':output.to(self.device), 'id':self.data['source_id'][idx]}

    
class GaiaXPlabel_v2():
    """Gaia DR3 XP spectrum to stellar labels instance
    """
    def __init__(self, npydata, total_num=6000, part_train=True, device=torch.device('cpu')) -> None:
        self.data = np.load(npydata, allow_pickle=True).item()
        self.spec = self.data['norm_spec']
        self.total_num = total_num
        self.part_train = part_train
        self.device = device
    
    def __len__(self) -> int:
        if self.part_train:
            num_sets = self.total_num
        else:
            num_sets = len(self.spec)
        return num_sets

    def __getitem__(self, idx: int):

        lnflux = self.spec[idx,:]
        lnflux[np.isnan(lnflux)] = np.mean(lnflux)
        photo = np.vstack([self.data['J'][idx], 
                           self.data['H'][idx], 
                           self.data['K'][idx]]).reshape(-1)
        
        lnflux = torch.tensor(
            np.hstack([lnflux, photo]).reshape(1,-1).astype(np.float32)
        )

        abundance = np.vstack([self.data['moh'][idx], self.data['aom'][idx]])
        
        output = torch.tensor(abundance.reshape(1,-1).astype(np.float32))
        return {'x':lnflux.to(self.device), 'y':output.to(self.device), 'id':self.data['source_id'][idx]}

class GaiaXPlabel_cont():
    """Gaia DR3 XP continuous spectrum to stellar labels instance
    """
    def __init__(self, npydata, total_num=6000, part_train=True, device=torch.device('cpu')) -> None:
        self.data = np.load(npydata, allow_pickle=True).item()
        self.bp = self.data['norm_bp_coef']
        self.rp = self.data['norm_rp_coef']
        self.df = self.data['df']
        self.total_num = total_num
        self.part_train = part_train
        self.device = device
    
    def __len__(self) -> int:
        if self.part_train:
            num_sets = self.total_num
        else:
            num_sets = len(self.df)
        return num_sets

    def __getitem__(self, idx: int):

        coeffs = np.hstack([self.bp[idx], self.rp[idx]])
        coeffs[np.isnan(coeffs)] = np.mean(coeffs)
        photo = self.df[['J','H','K']].values[idx]
        
        coeffs = torch.tensor(
            np.hstack([coeffs, photo]).reshape(1,-1).astype(np.float32)
        )

        abundance = self.df[['M_H','ALPHA_M']].values[idx]
        e_abundance = self.df[['M_H_ERR', 'ALPHA_M_ERR']].values[idx]
        output   = torch.tensor(abundance.reshape(-1).astype(np.float32))
        e_output = torch.tensor(e_abundance.reshape(-1).astype(np.float32))

        return {'x':coeffs.to(self.device), 'y':output.to(self.device), 'e_y':e_output.to(self.device), 'id':self.df['source_id'].values[idx]}


class GaiaXPlabel_cont_infer():
    """Gaia DR3 XP continuous spectrum to stellar labels instance
    """
    def __init__(self, npydata, total_num=6000, part_train=True, device=torch.device('cpu')) -> None:
        self.data = np.load(npydata, allow_pickle=True).item()
        self.bp = self.data['norm_bp_coef']
        self.rp = self.data['norm_rp_coef']
        self.df = self.data['df']
        self.total_num = total_num
        self.part_train = part_train
        self.device = device
    
    def __len__(self) -> int:
        if self.part_train:
            num_sets = self.total_num
        else:
            num_sets = len(self.df)
        return num_sets

    def __getitem__(self, idx: int):

        coeffs = np.hstack([self.bp[idx], self.rp[idx]])
        coeffs[np.isnan(coeffs)] = np.mean(coeffs)
        photo = self.df[['Jmag','Hmag','Kmag']].values[idx]
        
        coeffs = torch.tensor(
            np.hstack([coeffs, photo]).reshape(1,-1).astype(np.float32)
        )

        return {'x':coeffs.to(self.device), 'id':self.df['source_id'].values[idx]}