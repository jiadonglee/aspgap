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



class GaiaXPlabel_cont_v0():
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
        self.teff_max = self.df['TEFF'].max()
        self.J_max  = self.df['J'].max()
    
    def __len__(self) -> int:
        if self.part_train:
            num_sets = self.total_num
        else:
            num_sets = len(self.df)
        return num_sets

    def __getitem__(self, idx: int):

        coeffs = np.hstack([self.bp[idx][:8], self.rp[idx][:8]])
        coeffs[np.isnan(coeffs)] = np.mean(coeffs)
        photo = self.df[['J','H','K']].values[idx]/self.J_max
        
        coeffs = torch.tensor(
            np.hstack([coeffs, photo]).reshape(1,-1).astype(np.float32)
        )

        output = self.df[['TEFF', 'LOGG', 'M_H','ALPHA_M']].values[idx]
        e_output = self.df[['TEFF_ERR', 'LOGG_ERR', 'M_H_ERR', 'ALPHA_M_ERR']].values[idx]
        
        e_output[0]= e_output[0]/self.teff_max
        output[0]  = output[0]/self.teff_max

        output   = torch.tensor(output.reshape(-1).astype(np.float32))
        e_output = torch.tensor(e_output.reshape(-1).astype(np.float32))

        return {'x':coeffs.to(self.device), 'y':output.to(self.device), 'e_y':e_output.to(self.device), 'id':self.df['source_id'].values[idx]}    
    
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
        self.teff_max = self.df['TEFF'].max()
        self.J_max  = self.df['J'].max()
    
    def __len__(self) -> int:
        if self.part_train:
            num_sets = self.total_num
        else:
            num_sets = len(self.df)
        return num_sets

    def __getitem__(self, idx: int):

        coeffs = np.hstack([self.bp[idx][:14], self.rp[idx][:13]])
        coeffs[np.isnan(coeffs)] = np.mean(coeffs)
        photo = self.df[['J','H','K']].values[idx]/self.J_max
        
        coeffs = torch.tensor(
            np.hstack([coeffs, photo]).reshape(1,-1).astype(np.float32)
        )

        output = self.df[['TEFF', 'LOGG', 'M_H','ALPHA_M']].values[idx]
        e_output = self.df[['TEFF_ERR', 'LOGG_ERR', 'M_H_ERR', 'ALPHA_M_ERR']].values[idx]
        
        e_output[0]= e_output[0]/self.teff_max
        output[0]  = output[0]/self.teff_max

        output   = torch.tensor(output.reshape(-1).astype(np.float32))
        e_output = torch.tensor(e_output.reshape(-1).astype(np.float32))

        return {'x':coeffs.to(self.device), 'y':output.to(self.device), 'e_y':e_output.to(self.device), 'id':self.df['source_id'].values[idx]}


class GaiaXPlabel_forcast():
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
        self.teff_max = self.df['TEFF'].max()
        self.J_max  = self.df['J'].max()
    
    def __len__(self) -> int:
        if self.part_train:
            num_sets = self.total_num
        else:
            num_sets = len(self.df)
        return num_sets

    def __getitem__(self, idx: int):

        coeffs = np.hstack([self.bp[idx][:14], self.rp[idx][:13]])
        coeffs[np.isnan(coeffs)] = np.mean(coeffs)
        photo = self.df[['J','H','K']].values[idx]/self.J_max
        
        coeffs = torch.tensor(
            np.hstack([coeffs, photo]).reshape(1,-1).astype(np.float32)
        )
        output = self.df[['TEFF', 'LOGG', 'M_H','ALPHA_M']].values[idx]
        e_output = self.df[['TEFF_ERR', 'LOGG_ERR', 'M_H_ERR', 'ALPHA_M_ERR']].values[idx]
        
        e_output[0]= e_output[0]/self.teff_max
        output[0]  = output[0]/self.teff_max

        tgt = torch.tensor(np.hstack([photo[-1], output[:-1]]))
        
        output   = torch.tensor(output.reshape(-1).astype(np.float32))
        e_output = torch.tensor(e_output.reshape(-1).astype(np.float32))

        return {'x':coeffs.to(self.device), 'y':output.to(self.device), 'e_y':e_output.to(self.device), 'tgt':tgt.to(self.device), 'id':self.df['source_id'].values[idx]}


    
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


class GaiaXPlabel_cont_norm():
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
        self.J_max  = self.df['J'].max()
    
    def __len__(self) -> int:
        if self.part_train:
            num_sets = self.total_num
        else:
            num_sets = len(self.df)
        return num_sets

    def __getitem__(self, idx: int):

        coeffs = np.hstack([self.bp[idx][:14], self.rp[idx][:13]])
        coeffs[np.isnan(coeffs)] = np.mean(coeffs)
        photo = self.df[['J','H','K']].values[idx]/self.J_max
        
        coeffs = torch.tensor(
            np.hstack([coeffs, photo]).reshape(1,-1).astype(np.float32)
        ).to(self.device)

        output = self.df[['TEFF', 'LOGG', 'M_H','ALPHA_M']].values[idx]
        e_output = self.df[['TEFF_ERR', 'LOGG_ERR', 'M_H_ERR', 'ALPHA_M_ERR']].values[idx]

        """
        normalize stellar labels
        raw: 
        Teff [3000-6500] (3500), Logg [0-5] (5), [M/H] [-2-0.5] (2.5)
        [a/M] [-0.2-0.4] (0.6)

        after normalization:
        (Teff, Logg, [M/H], [a/M]) => (35, 25, 25, 30)
        """
        label_norm = torch.tensor(np.array([1e-2, 5., 10., 50.]).astype(np.float32))
        # output[0], e_output[0]= output[0]*1e-2, e_output[0]*1e-2
        # output[1], e_output[1]= output[1]*5., e_output[1]*5.
        # output[2], e_output[2]= output[2]*10., e_output[2]*10.
        # output[3], e_output[3]= output[3]*50., e_output[3]*50.
        output   = torch.tensor(output.reshape(-1).astype(np.float32)).to(self.device)
        e_output = torch.tensor(e_output.reshape(-1).astype(np.float32)).to(self.device)

        output   *= label_norm.to(self.device)
        e_output *= label_norm.to(self.device)

        return {'x':coeffs, 'y':output, 'e_y':e_output, 'id':self.df['source_id'].values[idx]}


class GaiaXP_allcoefs_label_cont_norm():
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
        self.J_max  = self.df['J'].max()
    
    def __len__(self) -> int:
        if self.part_train:
            num_sets = self.total_num
        else:
            num_sets = len(self.df)
        return num_sets

    def __getitem__(self, idx: int):

        coeffs = np.hstack([self.bp[idx], self.rp[idx]])
        coeffs[np.isnan(coeffs)] = np.mean(coeffs)
        photo = self.df[['J','H','K']].values[idx]/self.J_max
        
        coeffs = torch.tensor(
            np.hstack([coeffs, photo]).reshape(1,-1).astype(np.float32)
        ).to(self.device)

        output = self.df[['TEFF', 'LOGG', 'M_H','ALPHA_M']].values[idx]
        e_output = self.df[['TEFF_ERR', 'LOGG_ERR', 'M_H_ERR', 'ALPHA_M_ERR']].values[idx]

        """
        normalize stellar labels
        raw: 
        Teff [3000-6500] (3500), Logg [0-5] (5), [M/H] [-2-0.5] (2.5)
        [a/M] [-0.2-0.4] (0.6)

        after normalization:
        (Teff, Logg, [M/H], [a/M]) => (35, 25, 25, 30)
        """
        label_norm = torch.tensor(np.array([1e-2, 5., 10., 50.]).astype(np.float32))
        output   = torch.tensor(output.reshape(-1).astype(np.float32)).to(self.device)
        e_output = torch.tensor(e_output.reshape(-1).astype(np.float32)).to(self.device)

        output   *= label_norm.to(self.device)
        e_output *= label_norm.to(self.device)

        return {'x':coeffs, 'y':output, 'e_y':e_output, 'id':self.df['source_id'].values[idx]}

    
    
class GaiaXP_allcoefs_5label_cont_norm():
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
        self.J_max  = self.df['J'].max()
    
    def __len__(self) -> int:
        if self.part_train:
            num_sets = self.total_num
        else:
            num_sets = len(self.df)
        return num_sets

    def __getitem__(self, idx: int):

        coeffs = np.hstack([self.bp[idx], self.rp[idx]])
        coeffs[np.isnan(coeffs)] = np.mean(coeffs)
        photo = self.df[['J','H','K']].values[idx]/self.J_max
        
        coeffs = torch.tensor(
            np.hstack([coeffs, photo]).reshape(1,-1).astype(np.float32)
        ).to(self.device)

        output = self.df[['TEFF', 'LOGG', 'GAIAEDR3_PARALLAX', 'M_H','ALPHA_M']].values[idx]
        e_output = self.df[['TEFF_ERR', 'LOGG_ERR', 'GAIAEDR3_PARALLAX_ERROR', 'M_H_ERR', 'ALPHA_M_ERR']].values[idx]

        """
        normalize stellar labels
        raw: 
        Teff [3000-6500] (3500), Logg [0-5] (5), 
        parallax [-0.1-2] (2)
        [M/H]    [-2-0.5] (2.5)
        [a/M]    [-0.2-0.4] (0.6)

        after normalization:
        (Teff, Logg, [M/H], [a/M]) => (35, 25, 30, 25, 30)
        """
        label_norm = torch.tensor(np.array([1e-2, 5., 15., 10., 50.]).astype(np.float32))
        output   = torch.tensor(output.reshape(-1).astype(np.float32)).to(self.device)
        e_output = torch.tensor(e_output.reshape(-1).astype(np.float32)).to(self.device)

        output   *= label_norm.to(self.device)
        e_output *= label_norm.to(self.device)

        return {'x':coeffs, 'y':output, 'e_y':e_output, 'id':self.df['source_id'].values[idx]}


class GaiaXP_55coefs_5label_cont_ANDnorm():
    """Gaia DR3 XP continuous spectrum to stellar labels instance
    """
    def __init__(self, npydata, total_num=6000, part_train=True, device=torch.device('cpu')) -> None:
        self.data = np.load(npydata, allow_pickle=True).item()
        self.bp = self.data['lgnorm_bp_andrae']
        self.rp = self.data['lgnorm_rp_andrae']
        self.df = self.data['df']
        self.total_num = total_num
        self.part_train = part_train
        self.device = device
        self.J_max  = self.df['J'].max()
    
    def __len__(self) -> int:
        if self.part_train:
            num_sets = self.total_num
        else:
            num_sets = len(self.df)
        return num_sets

    def __getitem__(self, idx: int):

        coeffs = np.hstack([self.bp[idx], self.rp[idx]])
        coeffs[np.isnan(coeffs)] = np.mean(coeffs)
        photo = self.df[['J','H','K']].values[idx]/self.J_max
        
        coeffs = torch.tensor(
            np.hstack([coeffs, photo]).reshape(1,-1).astype(np.float32)
        ).to(self.device)

        output = self.df[['TEFF', 'LOGG', 'GAIAEDR3_PARALLAX', 'M_H','ALPHA_M']].values[idx]
        e_output = self.df[['TEFF_ERR', 'LOGG_ERR', 'GAIAEDR3_PARALLAX_ERROR', 'M_H_ERR', 'ALPHA_M_ERR']].values[idx]

        """
        normalize stellar labels
        raw: 
        Teff [3000-6500] (3500), 
        Logg [0-5] (5), 
        parallax [-0.1-2] (2)
        [M/H]    [-2-0.5] (2.5)
        [a/M]    [-0.2-0.4] (0.6)
        after normalization:
        (Teff, Logg, parallax, [M/H], [a/M]) => (35, 25, 30, 25, 30)
        """
        label_norm = torch.tensor(np.array([1e-2, 5., 15., 10., 50.]).astype(np.float32))
        output   = torch.tensor(output.reshape(-1).astype(np.float32)).to(self.device)
        e_output = torch.tensor(e_output.reshape(-1).astype(np.float32)).to(self.device)

        output   *= label_norm.to(self.device)
        e_output *= label_norm.to(self.device)

        return {'x':coeffs, 'y':output, 'e_y':e_output, 'id':self.df['source_id'].values[idx]}


class GXP_5lb():
    """Gaia DR3 XP continuous spectrum to stellar labels instance
    """
    def __init__(self, data, total_num=6000, part_train=True, device=torch.device('cpu')) -> None:
        
        self.data = np.load(data, allow_pickle=True).item()
        self.xp = self.data['norm_xp']
        self.labels = self.data['labels']
        self.e_labels = self.data['e_labels']
        self.source_id = self.data['source_id']
        self.total_num = total_num
        self.part_train = part_train
        self.device = device
    
    def __len__(self) -> int:
        if self.part_train:
            num_sets = self.total_num
        else:
            num_sets = len(self.labels)
        return num_sets

    def __getitem__(self, idx: int):
        x = torch.tensor(self.xp[idx].astype(np.float32)).to(self.device)
        y = torch.tensor(self.labels[idx].astype(np.float32)).to(self.device)
        e_y = torch.tensor(self.e_labels[idx].astype(np.float32)).to(self.device)
        return {'x':x, 'y':y, 'e_y':e_y, 'id':self.source_id[idx]}


class GXP_5lb_infer():
    """Gaia DR3 XP continuous spectrum to stellar labels instance
    """
    def __init__(self, data, device=torch.device('cpu')) -> None:
        
        if isinstance(data, str):
            self.data = np.load(data, allow_pickle=True).item()
        else:
            self.data = data

        self.xp = self.data['norm_xp']
        self.source_id = self.data['source_id']
        self.device = device
    
    def __len__(self) -> int:
        num_sets = len(self.source_id)
        return num_sets

    def __getitem__(self, idx: int):
        x = torch.tensor(self.xp[idx].astype(np.float32)).to(self.device)
        return {'x':x, 'id':self.source_id[idx]}


class GXP_prlx():
    """Gaia DR3 XP continuous spectrum to stellar labels instance
    """
    def __init__(self, data, total_num=6000, part_train=True, device=torch.device('cpu')) -> None:
        
        self.data = np.load(data, allow_pickle=True).item()
        self.xp = self.data['norm_xp']
        self.labels = self.data['labels']
        self.e_labels = self.data['e_labels']
        self.source_id = self.data['source_id']
        self.total_num = total_num
        self.part_train = part_train
        self.device = device
    
    def __len__(self) -> int:
        if self.part_train:
            num_sets = self.total_num
        else:
            num_sets = len(self.labels)
        return num_sets

    def __getitem__(self, idx: int):
        x = torch.tensor(self.xp[idx].astype(np.float32)).to(self.device)
        y = torch.tensor(self.labels[idx].astype(np.float32)).to(self.device)
        e_y = torch.tensor(self.e_labels[idx].astype(np.float32)).to(self.device)
        return {'x':x, 'y':y, 'e_y':e_y, 'id':self.source_id[idx]}
    
    
class GXP_AP_4lb():
    """Gaia DR3 XP continuous spectrum to stellar labels instance
    """
    def __init__(self, data, total_num=6000, part_train=True, device=torch.device('cpu')) -> None:
        
        self.data = np.load(data, allow_pickle=True).item()
        self.xp = self.data['xp_ap']
        self.labels = self.data['labels']
        self.e_labels = self.data['e_labels']
        self.source_id = self.data['source_id']
        self.total_num = total_num
        self.part_train = part_train
        self.device = device
    
    def __len__(self) -> int:
        if self.part_train:
            num_sets = self.total_num
        else:
            num_sets = len(self.labels)
        return num_sets

    def __getitem__(self, idx: int):

        x = torch.tensor(self.xp[idx], dtype=torch.float32, device=self.device)
        
        y = torch.tensor(self.labels[idx], dtype=torch.float32, device=self.device)

        e_y = torch.tensor(self.e_labels[idx], 
         dtype=torch.float32, device=self.device)

        return {'x':x, 'y':y, 'e_y':e_y, 'id':self.source_id[idx]}