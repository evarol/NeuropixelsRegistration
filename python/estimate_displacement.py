#################################
# register raw data
# input: raw data
# output: registered raw data: float 32
#################################

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, notebook
import torch
import pickle
import os

from scipy.signal import butter, filtfilt
from numpy.fft import fft2, ifft2, fftshift, ifftshift # Python DFT
import pywt

from skimage.restoration import denoise_nl_means, estimate_sigma

from scipy.stats import norm
from scipy.ndimage import shift, gaussian_filter
from scipy.stats import zscore
from scipy.spatial.distance import cdist

from scipy.interpolate import griddata

from utils import merge_filtered_files

# function that wraps around all functions
# reader: yass reader
def estimate_displacement(reader, geomarray, 
             detection_threshold=6, 
             num_chans_per_spike=4,
             do_destripe=True,
             do_denoise=True,
             do_subsampling=True,
             reg_win_num=5,
             reg_block_num=10,
             iteration_num=4,
             vert_smooth=3,
             horz_smooth=7,
             reader_type='yass', # also has an option 'spikeglx'
             save_raster_info=True):
    
    # spike detection: detection + deduplication
    spike_output_directory = os.path.join('.', "spikes")
    if not os.path.exists(spike_output_directory):
        os.makedirs(spike_output_directory)

    # ts: a raw data batch of size (sampling frequency, n_channels)
    if reader_type == 'yass':
        n_batches = reader.n_batches
        for i in tqdm(range(n_batches)):
            ts = reader.read_data_batch(i)
            run_spike_detect(ts, geomarray, spike_output_directory, i, threshold=detection_threshold)
    elif reader_type == 'spikeglx':
        sf = int(reader.fs)
        n_batches = int(reader.ns / reader.fs) # recording length in seconds
        for i in tqdm(range(n_batches)):
            ts_memmap = reader._raw[sf*i:sf*(i+1),:-1]
            ts = np.empty(ts_memmap.shape, dtype=np.int16)
            ts[:] = ts_memmap
            run_spike_detect(ts, geomarray, spike_output_directory, i, threshold=detection_threshold)
    elif reader_type == 'None': # reader is directory to bin file
        second_bytesize = 2 * 385 * 30000
        n_batches = int(os.path.getsize(reader) / second_bytesize)
        for i in tqdm(range(n_batches)):
            ts = np.fromfile(reader, dtype=np.int16, count=385*30000, offset=385*30000*i)
            ts = ts.reshape((30000,-1))[:,:-1]
            run_spike_detect(ts, geomarray, spike_output_directory, i, threshold=detection_threshold)
        
    # generate raster
    depths, times, amps, widths = gen_raster_info(spike_output_directory, num_chans=num_chans_per_spike)
    if save_raster_info:
        # save depths, times, amps, widths
        np.save('depths.npy', depths)
        np.save('times.npy', times)
        np.save('amps.npy', amps)
        np.save('widths.npy', widths)
        
    raster = gen_raster(depths, times, amps, geomarray)
    
    if do_destripe:
        destriped = destripe(raster)
    else:
        destriped = raster
    
    if do_denoise:
        denoised = cheap_anscombe_denoising(destriped)
    else:
        denoised = destriped
    
    # decentralized registration
    total_shift = decentralized_registration(denoised, 
                                             win_num=reg_win_num,  # change for non-rigid registration
                                             reg_block_num=reg_block_num,  # change for non-rigid registration
                                             iter_num=iteration_num,
                                             vert_smooth=vert_smooth,
                                             horz_smooth=horz_smooth)
    
    np.save('total_shift.npy', total_shift)
    
    return total_shift


# function that checks raster plot
# reader: yass reader
def check_raster(reader, geomarray,
             dtype = "int16",
             detection_threshold=6, 
             num_chans_per_spike=4,
             do_destripe=True,
             do_denoise=True,
             reader_type='yass', # also has an option 'spikeglx'
             save_raster_info=True):
    
    # spike detection: detection + deduplication
    spike_output_directory = os.path.join('.', "spikes")
    if not os.path.exists(spike_output_directory):
        os.makedirs(spike_output_directory)

    # ts: a raw data batch of size (sampling frequency, n_channels)
    if reader_type == 'yass':
        n_batches = reader.n_batches
        for i in tqdm(range(n_batches)):
            ts = reader.read_data_batch(i)
            run_spike_detect(ts, geomarray, spike_output_directory, i, threshold=detection_threshold)
    elif reader_type == 'spikeglx':
        sf = int(reader.fs)
        n_batches = int(reader.ns / reader.fs) # recording length in seconds
        for i in tqdm(range(n_batches)):
            ts_memmap = reader._raw[sf*i:sf*(i+1),:-1]
            ts = np.empty(ts_memmap.shape, dtype=np.int16)
            ts[:] = ts_memmap
            run_spike_detect(ts, geomarray, spike_output_directory, i, threshold=detection_threshold)
    elif reader_type == 'None': # reader is directory to bin file
        if dtype == 'int16':
            second_bytesize = 2 * 385 * 30000
        elif dtype == 'float32':
            second_bytesize = 4 * 384 * 30000
        n_batches = int(os.path.getsize(reader) / second_bytesize)
        for i in tqdm(range(n_batches)):
            if dtype == 'int16':
                ts = np.fromfile(reader, dtype=np.int16, count=385*30000, offset=385*30000*i)
            elif dtype == 'float32':
                ts = np.fromfile(reader, dtype=np.float32, count=384*30000, offset=384*30000*i)
            ts = ts.reshape((30000,-1))[:,:-1]
            run_spike_detect(ts, geomarray, spike_output_directory, i, threshold=detection_threshold)
        
    # generate raster
    depths, times, amps, widths = gen_raster_info(spike_output_directory, num_chans=num_chans_per_spike)
    if save_raster_info:
        # save depths, times, amps, widths
        np.save('depths.npy', depths)
        np.save('times.npy', times)
        np.save('amps.npy', amps)
        np.save('widths.npy', widths)
        
    raster = gen_raster(depths, times, amps, geomarray)
    
    if do_destripe:
        destriped = destripe(raster)
    else:
        destriped = raster
    
    if do_denoise:
        denoised = cheap_anscombe_denoising(destriped)
    else:
        denoised = destriped
    
    return denoised


def register(reader, geomarray, total_shift,
             registration_interp='linear', # also has an option 'gpr'
             reader_type='yass',
             registration_type='non_rigid'):
    
    # ts: a raw data batch of size (sampling frequency, n_channels)
    if reader_type == 'yass':
        n_batches = reader.n_batches
    elif reader_type == 'spikeglx':
        n_batches = int(reader.ns / reader.fs) # recording length in seconds
    elif reader_type == 'None': # reader is directory to bin file
        second_bytesize = 2 * 385 * 30000
        n_batches = int(os.path.getsize(reader) / second_bytesize)
    
    # register raw data
    registered_output_directory = os.path.join('.', "registered")
    if not os.path.exists(registered_output_directory):
        os.makedirs(registered_output_directory)
        
    register_data(reader, 
                  total_shift, 
                  geomarray, 
                  registered_output_directory, 
                  registration_interp,
                  n_batches,
                  reader_type,
                  registration_type)
    
    merge_filtered_files(registered_output_directory, registered_output_directory, delete = True)

# detects + deduplicates spikes
def spike_detect(X, geom, radius, timeradius):
    spike = {}
    spike_chans = {}
    spike_times = {}
    spike_amps = {}
    spike_central_coor = {}
    spike_central_time = {}
    spike_central_amplitude = {}
    spike_central_chan = {}
    
    # connected spike event grabbing
    nonzero = np.nonzero(X)
    I, T = nonzero[0], nonzero[1]
    V = X[nonzero]

    t = 0
    while I.shape[0] > 0:
        t += 1
        idx = np.where(np.abs(T - T[0]) <  timeradius)
        #idx2 = np.where(np.linalg.norm(geom[I[idx]] - geom[I[0]], axis=1) < radius)
        idx2 = np.where(cdist(geom[I[idx]], geom[I[0]][np.newaxis,:], 'euclidean')[:,0] < radius)
        
        spike_chans[t] = I[idx[0][idx2]]
        spike_times[t] = T[idx[0][idx2]]
        spike_amps[t] = V[idx[0][idx2]]

        I = np.delete(I, idx[0][idx2])
        T = np.delete(T, idx[0][idx2])
        V = np.delete(V, idx[0][idx2])
    # print('spike event grabbing done!')

    # de-duplication
    for t in spike_chans.keys():
        spike_central_coor[t] = 0
        spike_central_time[t] = 0

        r = (spike_amps[t]/spike_amps[t].sum())[:,np.newaxis]
        spike_central_coor[t] += (geom[spike_chans[t]]*r).sum(0)
        spike_central_time[t] += (spike_times[t]*r[:,0]).sum()

        spike_central_amplitude[t] = spike_amps[t].max()
        #idx = np.linalg.norm(geom - spike_central_coor[t], axis=1).argmin()
        idx = cdist(geom, spike_central_coor[t][np.newaxis,:], 'euclidean')[:,0].argmin()
        spike_central_chan[t] = idx

    spike['chans'] = spike_chans
    spike['times'] = spike_times
    spike['amps'] = spike_amps
    spike['central_coor'] = spike_central_coor
    spike['central_time'] = spike_central_time
    spike['central_amplitude'] = spike_central_amplitude
    spike['central_chan'] = spike_central_chan
    
    return spike

# butterworth filtering
def butterworth_filtering(ts, low_frequency=300, high_frequency=2000, order=3, sampling_frequency=30000):
    low = float(low_frequency) / sampling_frequency * 2
    high = float(high_frequency) / sampling_frequency * 2
    b,a = butter(order, [low, high], btype='bandpass', analog=False)
    
    T,C = ts.shape
    output = np.zeros((T,C), 'float32')
    for c in range(C):
        output[:, c] = filtfilt(b, a, ts[:,c])
       
    return output

def run_spike_detect(ts, geom, output_directory, batch_id, threshold=6, sf=30000,
                     low_frequency=300, high_frequency=2000, order=3,
                     spatial_radius=100, time_radius=100, decorr_iter=2):
    
    if os.path.exists(os.path.join(
        output_directory, "spike_{}.pkl".format(str(batch_id).zfill(6)))):
        return
    mp2d= torch.nn.MaxPool2d(kernel_size = [25, 1], stride = [2,1]).cuda()
    ts = butterworth_filtering(ts, low_frequency, high_frequency, order, sf)
    for i in range(decorr_iter):
        ts = zscore(ts, axis=1)
        ts = zscore(ts, axis=0)
    
    ts = torch.from_numpy(ts).cuda().float()
    ptp_sliding = mp2d(ts[None])[0] + mp2d(-ts[None])[0]
    ptp_sliding = np.asarray(ptp_sliding.cpu().T)
    ptp_sliding[np.where(ptp_sliding <= threshold)] = 0
    
    # args: ptp_sliding, geomarray, spatial_radius, time_radius
    spike = spike_detect(ptp_sliding, geom, spatial_radius, time_radius)
    
    fname = os.path.join(output_directory, "spike_{}.pkl".format(str(batch_id).zfill(6)))

    with open(fname, 'wb') as f:
        pickle.dump(spike, f, protocol=pickle.HIGHEST_PROTOCOL)


def gen_raster_info(saved_directory, sf=30000, num_chans=4, delete=True):
    depths = []
    times = []
    amps = []
    widths = []
    
    filenames = os.listdir(saved_directory)
    filenames_sorted = sorted(filenames)
    
    for fname in tqdm(filenames_sorted):
        t = int(fname.split('_')[-1].rstrip('.pkl'))
        with open(os.path.join(saved_directory, fname), 'rb') as f:
            spike = pickle.load(f)
        for j in spike['chans'].keys():
            if np.unique(spike['chans'][j]).shape[0] > num_chans:
                depths.append(spike['central_coor'][j][1])
                widths.append(spike['central_coor'][j][0])
                times.append(t + spike['central_time'][j]/(sf/2))
                amps.append(spike['central_amplitude'][j])
        if delete==True:
            os.remove(os.path.join(saved_directory, fname))

    depths = np.asarray(depths)
    times = np.asarray(times)
    amps = np.asarray(amps)
    widths = np.asarray(widths)
    
    return depths, times, amps, widths

def gen_raster(depths, times, amps, geom):
    max_t = np.ceil(times.max()).astype(int)
    D = geom[:,1].max().astype(int)
    
    raster = np.zeros((D,max_t))
    raster_count = np.zeros((D,max_t))
    for i in tqdm(range(max_t)):
        idx = np.intersect1d(np.where(times > i)[0], np.where(times < i+1)[0])

        for j in idx:
            depth = int(np.floor(depths[j]))
            amp = amps[j]
            raster[depth,i] += amp
            raster_count[depth,i] += 1

    raster_count[np.where(raster_count == 0)] = 1
            
    return raster/raster_count

def cheap_anscombe_denoising(z, sigma=1, h=0.1, estimate_sig=True, fast_mode=True, multichannel=False):
    minmax = (z - z.min()) / (z.max() - z.min()) # scales data to 0-1
    
    # Gaussianizing Poissonian data
    z_anscombe = 2. * np.sqrt(minmax + (3. / 8.))
    
    if estimate_sig:
        sigma = np.mean(estimate_sigma(z_anscombe, multichannel=multichannel))
        print("estimated sigma: {}".format(sigma))
    # Gaussian denoising
    z_anscombe_denoised = denoise_nl_means(z_anscombe, h=h*sigma, sigma=sigma, fast_mode=fast_mode) # NL means denoising

    z_inverse_anscombe = (z_anscombe_denoised / 2.)**2 + 0.25 * np.sqrt(1.5) * z_anscombe_denoised**-1 - (11. / 8.) * z_anscombe_denoised**-2 +(5. / 8.) * np.sqrt(1.5) * z_anscombe_denoised**-3 - (1. / 8.)
    
    z_inverse_anscombe_scaled = ((z.max() - z.min()) * z_inverse_anscombe) + z.min()
    
    return z_inverse_anscombe_scaled

def destripe(raster):
    D, W = raster.shape
    LL0 = raster
    wlet = 'db5'
    coeffs = pywt.wavedec2(LL0, wlet)
    L = len(coeffs)
    for i in range(1,L):
        HL = coeffs[i][1]    
        Fb = fft2(HL)   
        Fb = fftshift(Fb)
        mid = Fb.shape[0]//2
        Fb[mid,:] = 0
        Fb[mid-1,:] /= 3
        Fb[mid+1,:] /= 3
        Fb = ifftshift(Fb)   
        coeffs[i]= (coeffs[i][0], np.real(ifft2(Fb)), coeffs[i][2] )
    LL = pywt.waverec2(coeffs, wlet)
    LL = LL[:D,:W]
    
    destriped = np.zeros_like(raster)
    destriped[:D,:W] = LL
    return destriped

def get_gaussian_window(height, width, loc, scale=1):
    window = np.zeros((height,width))
    for i in range(height):
        window[i] = norm.pdf(i, loc=loc, scale=scale)
    return window / window.max()

def calc_displacement_matrix_raster(raster, nbins=1, disp = 400, step_size = 1, batch_size = 1):
    T = raster.shape[0]
    possible_displacement = np.arange(-disp, disp + step_size, step_size)
    raster = torch.from_numpy(raster).cuda().float()
    c2d = torch.nn.Conv2d(in_channels = 1, out_channels = T, kernel_size = [nbins, raster.shape[-1]], stride = 1, padding = [0, possible_displacement.size//2], bias = False).cuda()
    c2d.weight[:,0] = raster
    displacement = np.zeros([T, T])
    for i in notebook.tqdm(range(T//batch_size)):
        res = c2d(raster[i*batch_size:(i+1)*batch_size,None])[:,:,0,:].argmax(2)
        displacement[i*batch_size:(i+1)*batch_size] = possible_displacement[res.cpu()]
        del res
    del c2d
    del raster
    torch.cuda.empty_cache()
    return displacement

def calc_displacement(displacement, n_iter = 1000):
    p = torch.zeros(displacement.shape[0]).cuda()
    displacement = torch.from_numpy(displacement).cuda().float()
    n_batch = displacement.shape[0]
    pprev = p.clone()
    for i in notebook.tqdm(range(n_iter)):
        repeat1 = p.repeat_interleave(n_batch).reshape((n_batch, n_batch))
        repeat2 = p.repeat_interleave(n_batch).reshape((n_batch, n_batch)).T
        mat_norm = displacement + repeat1 - repeat2
        p += 2*(torch.sum(displacement-torch.diag(displacement), dim=1) - (n_batch-1)*p)/torch.norm(mat_norm)
        del mat_norm
        del repeat1
        del repeat2
        if torch.allclose(pprev, p):
            break
        else:
            del pprev
            pprev = p.clone()
    disp = np.asarray(p.cpu())
    del p
    del pprev
    del displacement
    torch.cuda.empty_cache()
    return disp

def shift_x(x, shift_amt):
    shifted = np.zeros_like(x)
    for t in range(x.shape[1]):
        col = x[:,t]
        sh = shift_amt[t]
        shifted[:,t] = shift(col, sh)

    return shifted

def register_raster(raster, total_shift, blocks):
    raster_sh = np.zeros_like(raster)
    for k in notebook.tqdm(range(1, blocks.shape[0])):
        cur = blocks[k]
        prev = blocks[k-1]
        sh = np.mean(-total_shift[prev:cur], axis=0)
        roi = np.zeros_like(raster)
        roi[prev:cur] = raster[prev:cur]
        raster_sh += shift_x(roi, sh)
    return raster_sh

def save_registered_raster(raster_sh, i, output_directory):
    fname = os.path.join(output_directory, "raster_{}.png".format(str(i+1).zfill(6)))
    print('plotting...')
    plt.figure(figsize=(16, 10))
    plt.imshow(raster_sh, vmin=0, vmax=10, aspect="auto", cmap=plt.get_cmap('inferno'))
    plt.ylabel("depth", fontsize=16)
    plt.xlabel("time", fontsize=16)
    plt.savefig(fname,bbox_inches='tight')
    plt.close()
    
def decentralized_registration(raster, win_num=1, reg_block_num=1, iter_num=4, vert_smooth=3, horz_smooth=7):
    D, T = raster.shape
    
    # get windows
    window_list = []
    if win_num == 1:
        window_list.append(np.ones_like(raster))
    else:
        space = int(D//(win_num+1))
        locs = np.linspace(space, D-space, win_num, dtype=np.int32)
        for i in range(win_num):
            window = get_gaussian_window(D, T, locs[i], scale=D/(0.5*win_num))
            window_list.append(window)
    window_sum = np.sum(np.asarray(window_list), axis=0)

    shifts = np.zeros((win_num, T))
    total_shift = np.zeros_like(raster)
    
    raster_i = gaussian_filter(raster, sigma=(vert_smooth, horz_smooth), order=0)
    
    reg_block_num += 1
    blocks = np.linspace(0, D, reg_block_num, dtype=np.int64)
    
    output_directory = os.path.join('.', "decentralized_raster")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    save_registered_raster(raster, -1, output_directory)
    
    for i in notebook.tqdm(range(iter_num)):
        
        shift_amt = np.zeros_like(raster)
        
        for j, w in enumerate(window_list):
            w_raster = w*raster_i
            displacement_matrix = calc_displacement_matrix_raster((w_raster).T[:,np.newaxis,:])
            disp = calc_displacement(displacement_matrix)
            shift_amt += w * disp[np.newaxis,:]
            shifts[j] += disp
            
        total_shift += (shift_amt / window_sum)

        raster_sh = register_raster(raster, total_shift, blocks)
        
        raster_i = gaussian_filter(raster_sh, sigma=(vert_smooth, horz_smooth), order=0)
        
        save_registered_raster(raster_sh, i, output_directory)
        
    return total_shift

def register_data(reader, total_shift, geomarray, 
                  registered_output_directory, interp, 
                  n_batches, reader_type, 
                  registration_type='non_rigid'):
    D, T = total_shift.shape
    ys = geomarray[:,1]
    n_chans = ys.shape[0]
    
    if registration_type == 'non_rigid':
        win_num = np.unique(ys).shape[0]

        # get windows
        window_list = []
        for i in tqdm(range(n_chans)):
            window = get_gaussian_window(D, T, ys[i], scale=D/(0.5*win_num))
            window_list.append(window)

        estimated_displacement = np.zeros((n_chans, total_shift.shape[1]))
        for i in tqdm(range(n_chans)):
            window = window_list[i]
            w_disp = total_shift * window
            w_disp = w_disp.sum(0) / window.sum(0)
            estimated_displacement[i] = w_disp
            
    elif registration_type == 'rigid':
        estimated_displacement = np.zeros((n_chans, total_shift.shape[1]))
        for i in tqdm(range(n_chans)):
            estimated_displacement[i] = total_shift[0]
        
    for batch_id in tqdm(range(int(n_batches))):
        if interp == 'linear':
            register_data_linear(
                batch_id, 
                reader, 
                registered_output_directory, 
                estimated_displacement,
                geomarray,
                n_chans,
                reader_type
            )
        elif interp == 'gpr':
            pass


def register_data_linear(i, reader, registered_output_directory, estimated_displacement, geomarray, n_chans, reader_type):
    if reader_type == 'yass':
        ts = reader.read_data_batch(i)
    elif reader_type == 'spikeglx':
        sf = int(reader.fs)
        ts_memmap = reader._raw[sf*i:sf*(i+1),:-1]
        ts = np.empty(ts_memmap.shape, dtype=np.int16)
        ts[:] = ts_memmap
    elif reader_type == 'None': # reader is directory to bin file
        ts = np.fromfile(reader, dtype=np.int16, count=385*30000, offset=385*30000*i)
        ts = ts.reshape((30000,-1))[:,:-1]
    ts = ts.T
    disp = np.concatenate((np.zeros(n_chans)[:,None],estimated_displacement[:,i][:,None]), axis=1)
    ts = griddata(geomarray, ts, geomarray + disp, method = 'linear', fill_value = 0)
    np.save(os.path.join(
            registered_output_directory,
            "registered_{}.npy".format(
                str(i).zfill(6))), ts.T.astype(np.float32))
