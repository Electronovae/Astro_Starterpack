import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import table, coordinates, units, wcs, time
from astropy.table import Table

# Class pour faire la photométrie d'ouverture
class aphoto_modified:

    def __init__(self,fdata_r):
        img_r = fdata_r[0].data
        self.img_r_bkg = img_r - np.nanmedian(img_r)
        self.magzp_r = fdata_r[0].header['MAGZP'] 
        self.wcs_img_r = wcs.WCS(header=fdata_r[0].header)

    def aperture(self, x0, y0, radius):
        sky = self.wcs_img_r.pixel_to_world(x0, y0)
        ra = sky.ra.degree
        dec = sky.dec.degree
        # Image band 'r'
        xr, yr = self.wcs_img_r.world_to_pixel(sky)
        xrmin, xrmax = xr-3*radius, xr+3*radius
        yrmin, yrmax = yr-3*radius, yr+3*radius
        # Affichage des images
        fig = plt.figure(figsize=(5, 10))
        fig.add_subplot(211, title='Image avec le filtre r')
        plt.imshow(self.img_r_bkg,cmap='viridis', vmin=0, vmax=200) 
        circler = plt.Circle((xr,yr), radius, fill=False, edgecolor='red', linewidth=5) 
        plt.gcf().gca().add_artist(circler)
        plt.ylim(yrmin,yrmax) 
        plt.xlim(xrmin,xrmax)
        # Fin
        return

    def star_photometry(self, x0, y0, radius): 
        sky = self.wcs_img_r.pixel_to_world(x0, y0)
        ra = sky.ra.degree
        dec = sky.dec.degree         
        # Filtre r
        xr, yr = self.wcs_img_r.world_to_pixel(sky)
        x0r = round(xr.mean())
        y0r = round(yr.mean())
        flux_r = 0
        for xi in range(x0r-radius,x0r+radius,1):
            for yi in range(y0r-radius,y0r+radius,1):
                ri = np.sqrt((xi-x0r)**2 + (yi-y0r)**2)
                if (ri<radius): 
                    flux_r += self.img_r_bkg[yi,xi]
        if (flux_r<0):
            flux_r = 1
        mag_r = -2.5*(np.log10(flux_r)) + self.magzp_r
        # Fin
        data = (ra, dec, flux_r, mag_r)
        return data

    def photometry(self, list_star): 
        ra, dec, flux_r, mag_r = [], [], [], []
        for i in range(len(list_star)):
            x = list_star[i][0]
            y = list_star[i][1]
            r = list_star[i][2]
            data = self.star_photometry(x, y, r)
            ra.append(data[0])
            dec.append(data[1])
            flux_r.append(data[2])
            mag_r.append(data[3])
        tab = Table([ra,dec,flux_r,mag_r], names=('ra','dec','flux_r','mag_r'))
        return tab






import os
from astropy.wcs import WCS
from reproject import reproject_interp
import shutil
from scipy.stats import norm
from scipy.ndimage import gaussian_filter
from datetime import datetime, timedelta
from astropy.table import vstack
from matplotlib.patches import Rectangle
from astropy.coordinates import SkyCoord
import glob
import matplotlib.patches as patches
import warnings
from photutils.background import Background2D, MedianBackground
from astropy.stats import SigmaClip , sigma_clipped_stats
from astropy.time import Time
from scipy.ndimage import median_filter, zoom
import numpy.ma as ma
from photutils.detection import DAOStarFinder
from astropy.utils.exceptions import AstropyWarning
import math

class AstroTools:
    
    def __init__(self, fits_files, file_name ='00Field_Filter_cCcd_o_qQuadrant'):
        self.fits_files = fits_files 
        self.first_file = fits_files[0]
        self.first_hdu = fits.open(self.first_file)[0]
        self.first_wcs = WCS(self.first_hdu.header)
        self.first_shape = self.first_hdu.data.shape
        self.file_name = file_name
        




    def show_img(self, data, title='', vmin=None, vmax=None, cmap=None):
        plt.figure(figsize=(7, 7))
        im = plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label('Flux (ADU)', fontsize=14)
        plt.title(title, fontsize=16, pad=20)
        plt.xlabel('X pixel', fontsize=14)
        plt.ylabel('Y pixel', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()
    
    
    
    
    
    def radec_to_pixel(self, ra, dec, wcs):
        skycoord = SkyCoord(ra, dec, unit='deg')
        x, y = wcs.world_to_pixel(skycoord)
        return x, y
    
    
    
    
    
    def pixel_to_radec(self, x, y, wcs):
        skycoord = wcs.pixel_to_world(x, y)
        ra = skycoord.ra.degree
        dec = skycoord.dec.degree
        return ra, dec






    def get_zoom(self, image, ra , dec , wcs , zoom=50):
        x_pixel, y_pixel = self.radec_to_pixel(ra, dec, wcs)  
        x_min = max(int(x_pixel - zoom), 0)
        x_max = min(int(x_pixel + zoom), image.shape[1])
        y_min = max(int(y_pixel - zoom), 0)
        y_max = min(int(y_pixel + zoom), image.shape[0])
        return x_min , x_max , y_min , y_max






    def reproject_astropy(self, source_data, source_wcs, target_wcs, target_shape):
        array, _ = reproject_interp((source_data, source_wcs), target_wcs, shape_out=target_shape, order='bilinear')
        return array
    
    
    
    
    
    
    def degrade_psf(self, image, seeing_from, seeing_to, pixel_scale=1.01):
        """
        Applies a Gaussian filter to equalize the seeing of an image.
        - seeing_from: image seeing (in arcsec)
        - seeing_to: target seeing (larger, in arcsec)
        - pixel_scale: sample rate in arcsec/pixel
        """
        if seeing_to <= seeing_from:
            return image  # pas besoin de dégrader
        
        # Convertir seeing en sigma (pixels)
        fwhm_diff = np.sqrt(seeing_to**2 - seeing_from**2)
        sigma_pix = fwhm_diff / (2.3548 * pixel_scale)
        
        return gaussian_filter(image, sigma=sigma_pix)






    def star_photometry(self, x0, y0, radius, img, magzp):        
            x0 = round(x0.mean())
            y0 = round(y0.mean())
            flux = 0
            for xi in range(x0-radius,x0+radius,1):
                for yi in range(y0-radius,y0+radius,1):
                    ri = np.sqrt((xi-x0)**2 + (yi-y0)**2)
                    if (ri<radius): 
                        flux += img[yi,xi]
            if (flux<0):
                flux = 1
            mag = -2.5*(np.log10(flux)) + magzp
            # Fin
            data = Table(rows=[(flux, mag)], names=('flux', 'mag'))
            return data







    
    def remove_background_photutils(self, image, box_size=64, filter_size=(3, 3)):
        sigma_clip = SigmaClip(sigma=3.0)
        bkg_estimator = MedianBackground()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            bkg = Background2D(
                image,
                box_size=box_size,
                filter_size=filter_size,
                sigma_clip=sigma_clip,
                bkg_estimator=bkg_estimator
            )
        
        return image - bkg.background, bkg.background







    def remove_local_background(self, image, block_size=64):
        ny, nx = image.shape
        nx_blocks = nx // block_size
        ny_blocks = ny // block_size

        bkg_map = np.zeros((ny_blocks, nx_blocks))

        for i in range(ny_blocks):
            for j in range(nx_blocks):
                y0 = i * block_size
                y1 = y0 + block_size
                x0 = j * block_size
                x1 = x0 + block_size
                patch = image[y0:y1, x0:x1]

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    bkg_map[i, j] = np.nanmedian(patch)

        # Interpolation de la carte à la taille de l'image originale
        zoom_factor_y = ny / bkg_map.shape[0]
        zoom_factor_x = nx / bkg_map.shape[1]
        bkg_map_fullres = zoom(bkg_map, (zoom_factor_y, zoom_factor_x), order=1)
        image_bkg_sub = image - bkg_map_fullres
        
        return image_bkg_sub, bkg_map_fullres







    def distrib_seeing(self, fits_file=None , Plotshow = True):
        if fits_file is None:
            fits_file = self.fits_files
        else:
            fits_file = fits_file    
        seeing_values = []
        for file in fits_file:
            try:
                with fits.open(file) as hdu_list:
                    header = hdu_list[0].header
                    seeing = header['SEEING']
                    if seeing is not None:
                        seeing_values.append(seeing)
                    else:
                        print(f"SEEING not found for : {file}")
            except Exception as e:
                print(f"Error with {file} : {e}")

        mean_seeing = np.mean(seeing_values)
        median_seeing = np.median(seeing_values)
        sigma_seeing = np.std(seeing_values)
        seeing_max = np.max(seeing_values)
        x_vals = np.linspace(min(seeing_values), max(seeing_values), 1000)
        gaussian_curve = norm.pdf(x_vals, loc=mean_seeing, scale=sigma_seeing)* len(seeing_values)/10
        print(f"{len(seeing_values)} Seeing values found")
        print(f'seeing maximum = {seeing_max}')

        # Histogramme
        if Plotshow == True:
            plt.figure(figsize=(8, 5))
            plt.hist(seeing_values, bins=50, color='darkblue', edgecolor = 'orange')
            #plt.plot(x_vals, gaussian_curve, color='orange', linewidth=2, label='Gaussian distribution')
            plt.axvline(mean_seeing, color = 'r', label = f'mean seeing = {mean_seeing:.3f}',linewidth=3)
            plt.axvline(median_seeing, color = 'g', label = f'median seeing = {median_seeing:.3f}',linewidth=3)
            plt.axvline(mean_seeing+sigma_seeing, color = 'r',  linestyle = '--',linewidth=2)
            plt.axvline(mean_seeing-sigma_seeing, color = 'r',  linestyle = '--', label = f'sigma seeing = ± {sigma_seeing:.3f}',linewidth=2)
            plt.title(f"seeing distribution ({self.file_name}) | {len(seeing_values)} values",fontsize=14)
            plt.xlabel("Seeing",fontsize=14)
            plt.ylabel("Number of images",fontsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.tight_layout()
            plt.legend(fontsize=14)
            plt.show()
        return mean_seeing, median_seeing , sigma_seeing , seeing_max 







    def align_images(self, output_dir=None, show=False, max_plots=None):  # ~12 seconds/image
        if output_dir is None:
            output_dir = f"ztf_data_aligned\\{self.file_name}\\"
        os.makedirs(output_dir, exist_ok=True)

        # Charger le WCS de référence une seule fois
        with fits.open(self.first_file) as hdu_list:
            self.first_hdu = hdu_list[0]
            self.first_wcs = WCS(self.first_hdu.header)
            self.first_shape = self.first_hdu.data.shape
            self.first_wcs_header = self.first_wcs.to_header()

        numero = 0

        for fits_file in self.fits_files:
            filename = os.path.basename(fits_file)

            with fits.open(fits_file) as hdu_list:
                hdu = hdu_list[0]
                source_wcs = WCS(hdu.header)
                reprojected_data = self.reproject_astropy(hdu.data, source_wcs, self.first_wcs, self.first_shape)
                new_header = hdu.header.copy()
                new_header.update(self.first_wcs_header)
                new_header['NAXIS'] = 2
                new_header['NAXIS1'] = reprojected_data.shape[1]
                new_header['NAXIS2'] = reprojected_data.shape[0]

                aligned_filename = os.path.join(output_dir, f"aligned_{filename}")
                new_hdu = fits.PrimaryHDU(data=reprojected_data.astype(np.float32), header=new_header)
                new_hdu.writeto(aligned_filename, overwrite=True)

                
                print(f"Image aligned saved : {aligned_filename}")
                print(f"Image n° {numero}")

                if show and (max_plots is None or numero <= max_plots):
                    vmed = np.nanmedian(reprojected_data)
                    self.show_img(reprojected_data, f"Aligned {filename}", vmin=vmed, vmax=vmed + 100)
                    
                numero += 1
        print("Alignment completed for all valid images!")
        
        
        
        
        
        
        
        
               
        
    def get_refimg(self, aligned_files, ref_method='median', seeing_max=6, zp_ref=None, limit_bkg=400, Imshow_ref=True, Imshow=False, use_photutils_bkg=False, block_size=64, max_plots=None, stack_number=100, output_file='ztf_data_ref', n=''): 

        num_images = 0
        first_hdu_a = fits.open(aligned_files[0])[0]
        valid_paths = []
        dates = []
        
        
        if zp_ref is None:
            zp_ref = first_hdu_a.header['MAGZP']
        else: zp_ref = zp_ref
        print('zero point of magniture (ref) = ', zp_ref)
        
    
        if ref_method == 'mean':
            image_sum=np.zeros(first_hdu_a.data.shape, dtype=np.float32)
        if ref_method == 'median':
            temp_dir = f".\\temp_refimg_cache_{self.file_name}"
            os.makedirs(temp_dir, exist_ok=True)
            
            
        for fits_file in aligned_files:
            with fits.open(fits_file) as hdu_list:
                hdu = hdu_list[0]
                img_data = hdu.data
                print(fits_file)
                zp_img = hdu.header['MAGZP']
                seeing = hdu.header['SEEING']
                saturate = hdu.header['SATURATE']
                mjd = hdu.header['OBSMJD']
                sky_bkg = np.nanmedian(img_data)
                
                
                #==============================# FILTERING IMAGES #=============================#
                if use_photutils_bkg is False and sky_bkg > limit_bkg:
                    print(f"Image ignored because background too bright (clouds) : {fits_file}, median = {sky_bkg}")
                    continue
                if seeing > seeing_max:
                    print(f"Image ignored due to excessive seeing : {fits_file}, seeing = {seeing}")
                    continue
                
                num_images += 1
                
                #==============================# IMAGE CORRECTION #=============================#
                img_data[img_data >= saturate] = np.nan
                if use_photutils_bkg and sky_bkg > limit_bkg:
                    img_data_local, bkg_map = self.remove_background_photutils(img_data, box_size=block_size, filter_size=(5, 5))
                    print("Background removed using photutils")
                else:
                    img_data_local, bkg_map = self.remove_local_background(img_data, block_size=block_size)
                    
                img_data_corr = img_data_local * 10**((zp_ref - zp_img)/2.5)
                img_data_corr = self.degrade_psf(img_data_corr, seeing, seeing_max, pixel_scale=1.01)
                print('mag = ', zp_img)
                print ('seeing = ', seeing )
                print(f'n°{num_images}')
                
                #==============================# TEMPORARY SAVE IMAGES #=============================#
                if ref_method == 'median':
                    temp_path = os.path.join(temp_dir, f"img_{num_images:04d}.npy")
                    np.save(temp_path, img_data_corr.astype(np.float32))
                    valid_paths.append(temp_path)
                if ref_method == 'mean':
                    image_sum += np.array(img_data_corr, dtype=np.float32)
                    
                if num_images >= stack_number:
                    break    
                    
                #==============================# DATES AND FRACDAY #=============================#
                          
                file_date = Time(mjd, format='mjd').to_datetime()
                date_display = file_date.strftime('%Y-%m-%d')
                dates.append(date_display)
                fracday = f"{file_date.hour:02}{file_date.minute:02}{file_date.second:02}"
                
                #========================================# PLOTS #========================================#
                if Imshow and (max_plots is None or num_images <= max_plots):
                    _ , axes = plt.subplots(2, 2, figsize=(8, 8))
                    im = axes[0,0].imshow(img_data, vmin = np.nanmedian(img_data), vmax = np.nanmedian(img_data) + 30)
                    axes[0,0].set_title(f"{date_display} | fracday : {fracday}")
                    plt.colorbar(im, ax=axes[0,0], orientation='vertical', fraction=0.046, pad=0.04)
                    
                    axes[1,0].hist(img_data.flatten(), bins=100, alpha=0.5, color = 'b', label='Before correction')
                    axes[1,0].hist(img_data_corr.flatten(), bins=100, alpha=0.5, color = 'r', label='After correction')
                    axes[1,0].legend()
                    axes[1,0].set_title("Flux Histogram")
                    axes[1,0].set_xlabel("Flux")
                    axes[1,0].set_ylabel("Number of pixels")
                    axes[1,0].set_yscale('log')
                    
                    im0 = axes[0,1].imshow(img_data_corr, vmin=np.nanmedian(img_data_corr), vmax=np.nanmedian(img_data_corr) + 5)
                    axes[0,1].set_title(f"Corrected image : {date_display} | fracday : {fracday}")
                    plt.colorbar(im0, ax=axes[0,1], fraction=0.046, pad=0.04)
                    
                    """if use_photutils_bkg  :"""
                    im1 = axes[1,1].imshow(bkg_map, cmap='plasma', vmin=np.nanpercentile(bkg_map, 5), vmax=np.nanpercentile(bkg_map, 95))
                    plt.colorbar(im1, ax=axes[1,1], fraction=0.046, pad=0.04, label='Background level')
                    axes[1,1].set_title(f"Background {date_display} | fracday : {fracday}")
                    """else:
                        axes[1,1].text(0.5, 0.5, "No photutils background correction applied ", fontsize=12, ha='center', va='center')
                        axes[1,1].set_axis_off()"""
                    
                    plt.suptitle(f"Analysis : {os.path.basename(fits_file)}", fontsize=14)
                    plt.tight_layout()
                    plt.show()
                    
        if num_images == 0:
            raise ValueError("No valid image was selected to construct the reference image.")        
        #========================================# Median #========================================#
        if ref_method == 'median':
            sample = np.load(valid_paths[0])
            H, W = sample.shape
            image_median = np.empty((H, W), dtype=np.float32)
            print("Calculating the median line by line...")
            for i in range(H):
                rows = [np.load(p, mmap_mode='r')[i, :] for p in valid_paths]
                stack = np.vstack(rows)
                image_median[i, :] = np.median(stack, axis=0)
                
            output_path = f"{output_file}\\{self.file_name}\\"
            os.makedirs(output_path, exist_ok=True)
            output_filename = os.path.join(output_path, f"refimg_median_{self.file_name}.fits")
            new_hdu = fits.PrimaryHDU(data=image_median.astype(np.float32), header=first_hdu_a.header)
            new_hdu.header['SEEING'] = (round(float(seeing_max), 3), 'Mean seeing used to build image_median')
            new_hdu.header['NFRAMES'] = (num_images, 'Number of images used for image_median')
            new_hdu.writeto(output_filename, overwrite=True)
            print(num_images, 'Number of images used for image_median')
            
            if Imshow_ref == True:
                self.show_img(data=image_median, title="Reference Image (Median)", vmin = 0, vmax = 30)
            print(f"Recorded median reference image : {output_filename}")
            
            del rows
            shutil.rmtree(temp_dir)
        #========================================# Mean #========================================#
        if ref_method == 'mean':
            image_avg = image_sum / num_images
            output_path = f"{output_file}\\{self.file_name}\\"
            os.makedirs(output_path, exist_ok=True)
            output_filename = os.path.join(output_path, f"refimg_mean_{self.file_name}{n}.fits")
            new_hdu = fits.PrimaryHDU(data=np.array(image_avg), header=first_hdu_a.header)
            new_hdu.header['SEEING'] = (round(float(seeing_max), 3), 'Mean seeing used to build image_avg')
            new_hdu.header['NFRAMES'] = (num_images, 'Number of images used for image_avg')
            new_hdu.writeto(output_filename, overwrite=True)
            print(num_images, 'Number of images used for image_avg')
            
            if Imshow_ref == True:
                self.show_img(data=image_avg, title="Reference Image (Mean)", vmin = 0, vmax = 100)
            print(f"Recorded mean reference image : {output_filename}")
            
        
        
        
        
        
        
        
    def subtraction(self, aligned_files , ref_filename , ra , dec , Start_Year, 
                    Start_Month, Start_Day , End_Year, End_Month, End_Day, zp=None, limit_bkg=500, seeing_reference=None, Imshow=True, zoom=50 , aperture=15, use_photutils_bkg=False, block_size=64, max_plots=None): #seeing_max est donné par la fonction distrib_seeing()
        """ra    =    coordinated ra of the zoom
           dec   =    coordinated dec of the zoom"""
        num_images = 0
        ref_hdu = fits.open(ref_filename)[0]
        image_ref = ref_hdu.data
        image_ref -= np.nanmedian(image_ref)
        zp_ref = ref_hdu.header['MAGZP']
        seeing_ref = ref_hdu.header['SEEING']
        wcs_ref = WCS(ref_hdu.header)
        x_min_ref , x_max_ref , y_min_ref , y_max_ref = self.get_zoom(image_ref, ra , dec , wcs_ref , zoom=zoom)
        start_date = datetime(Start_Year, Start_Month, Start_Day)
        end_date = datetime(End_Year, End_Month, End_Day)
        start_mjd = Time(start_date).mjd
        end_mjd = Time(end_date).mjd
        fits_file_aligned_date_filtered = []
        dates = []
        fracday_list = []
        tab_data = []
        
        if zp is not None:
            image_ref = image_ref * 10**((zp - zp_ref)/2.5)
            zp_ref = zp
            
        if seeing_reference is not None:
            seeing_ref = seeing_reference
            
    #=============================# Filtering FITS files by dates #=============================#
        for file in aligned_files:
            try:
                with fits.open(file) as hdul:
                    mjd = hdul[0].header['OBSMJD']
                    if start_mjd <= mjd <= end_mjd:
                        fits_file_aligned_date_filtered.append(file)

            except Exception as e:
                print(f"Erreur lecture MJD dans {file} : {e}")
                continue
    #==========================================# SUBTRACTION #=======================================#    
        for fits_file in fits_file_aligned_date_filtered:
            with fits.open(fits_file) as hdu_list:
                hdu = hdu_list[0]
                img_data = hdu.data
                wcs = WCS(hdu.header)
                zp_img = hdu.header['MAGZP']
                seeing_img = hdu.header['SEEING']
                saturate = hdu.header['SATURATE']
                
                if use_photutils_bkg is False and np.nanmedian(img_data) > limit_bkg:
                    print(f"Image ignored for background too bright (clouds) : {fits_file}, median = {np.nanmedian(img_data)}")
                    continue
                if seeing_img > seeing_ref:
                    print(f"Image ignored for seeing out of range : {fits_file}, seeing = {seeing_img}")
                    continue
                #==============================# DATES AND FRACDAY #=============================#
                mjd = hdu.header['OBSMJD']
                file_date = Time(mjd, format='mjd').to_datetime()
                date_display = file_date.strftime('%Y-%m-%d')
                dates.append(date_display)

                # Extraire les heures/fracday
                fracday = f"{file_date.hour:02}{file_date.minute:02}{file_date.second:02}"
                fracday_list.append(fracday)
                #==============================# IMAGE CORRECTION #=============================#
                img_data[img_data >= saturate] = np.nan
                if use_photutils_bkg and np.nanmedian(img_data) > limit_bkg:
                    img_data_local, bkg_map = self.remove_background_photutils(img_data, box_size=64, filter_size=(5, 5))
                else:
                    img_data_local, bkg_map = self.remove_local_background(img_data, block_size=block_size)
                img_data_corr = img_data_local * 10**((zp_ref - zp_img)/2.5)
                
                if seeing_img < seeing_ref:
                    img_data_corr = self.degrade_psf(img_data_corr, seeing_img, seeing_ref)
                elif seeing_ref < seeing_img:
                    image_ref = self.degrade_psf(image_ref, seeing_ref, seeing_img)
                    
                num_images += 1
                diff_img = img_data_corr - image_ref
                diff_img -= np.nanmedian(diff_img)
                #x_pixel, y_pixel = self.radec_to_pixel(ra, dec, wcs)


                # ==============================# DETECTION TRANSIENTS AUTOMATIQUE #==============================
                transients = self.detect_transients(diff_img)

                if transients is not None and len(transients) > 0:
                    print(f"{len(transients)} transient(s) detected in {fits_file}")
                    
                    # Marquage d’un ID unique pour affichage
                    for idx, source in enumerate(transients):
                        x, y = source['xcentroid'], source['ycentroid']
                        phot = self.star_photometry(x, y, aperture, diff_img, zp_ref)
                        phot['date'] = date_display
                        phot['fracday'] = fracday
                        phot['x'] = x
                        phot['y'] = y
                        phot['id'] = idx  # ajout d’un ID temporaire pour visualisation
                        tab_data.append(phot)
                else:
                    print(f"No transient found in {fits_file}")
                    
                #==============================#  #=============================#
                x_min , x_max , y_min , y_max = self.get_zoom(img_data_corr, ra , dec , wcs , zoom=zoom)
                print ('SEEING = ', seeing_img )
                print ('Image n° ', num_images )
                print(date_display)

                #==============================# PLOTS #=============================#
                if Imshow and (max_plots is None or num_images <= max_plots):
                    
                    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
                    #-------------------#First Line#-------------------#
                    im0 = axes[1,0].imshow(img_data_corr[y_min:y_max, x_min:x_max], vmin=0, vmax=100) 
                    axes[1,0].set_title(f"{date_display} | fracday : {fracday} (zoom)", fontsize=16)
                    axes[1,0].tick_params(axis='both', labelsize=16)
                    x_center = (x_max - x_min) / 2
                    y_center = (y_max - y_min) / 2
                    circle = patches.Circle((x_center, y_center), radius=15, edgecolor='red', facecolor='none', linewidth=2, label=f'radius = {aperture} px')
                    axes[1,0].add_patch(circle)
                    axes[1,0].legend(loc='upper right', fontsize=16)
                    plt.colorbar(im0, ax=axes[1,0], fraction=0.046, pad=0.04)
                    
                    im1 = axes[1,1].imshow(image_ref[y_min_ref:y_max_ref, x_min_ref:x_max_ref], vmin=0, vmax=100)
                    axes[1,1].set_title("Ref img (zoom)", fontsize=16)
                    axes[1,1].tick_params(axis='both', labelsize=16)
                    plt.colorbar(im1, ax=axes[1,1], fraction=0.046, pad=0.04)           
                    
                    
                    im2 = axes[0, 0].imshow(img_data_corr, vmin=0, vmax=30)
                    axes[0,0].set_title(f"{date_display} | fracday : {fracday}", fontsize=16)
                    rect1 = Rectangle((x_min_ref, y_min_ref), x_max_ref - x_min_ref, y_max_ref - y_min_ref, linewidth=2, edgecolor='r', facecolor='none', label='Zoom (ref)')
                    axes[0,0].add_patch(rect1)
                    axes[0,0].legend(loc='upper right', fontsize=16)
                    axes[0,0].tick_params(axis='both', labelsize=16)
                    plt.colorbar(im2, ax=axes[0, 0], fraction=0.046, pad=0.04)
                    
                    
                    im3 = axes[0,1].imshow(image_ref, vmin=0, vmax=30)
                    axes[0,1].set_title("Ref img", fontsize=16)
                    axes[0,1].tick_params(axis='both', labelsize=16)
                    plt.colorbar(im3, ax=axes[0,1], fraction=0.046, pad=0.04)   
                    
                    
                            
                    #-------------------#Second Line#-------------------#
                    im4 = axes[1,2].imshow(diff_img[y_min_ref:y_max_ref, x_min_ref:x_max_ref], cmap='seismic', vmin=-50, vmax=50)
                    axes[1,2].set_title("Difference (Corrected - Ref.) (zoom)", fontsize=16)
                    axes[1,2].tick_params(axis='both', labelsize=16)
                    circle = patches.Circle((x_center, y_center), radius=15, edgecolor='red', facecolor='none', linewidth=2, label=f'radius = {aperture} px')
                    axes[1,2].add_patch(circle)
                    axes[1,2].legend(loc='upper right', fontsize=16)
                    plt.colorbar(im4, ax=axes[1,2], fraction=0.046, pad=0.04)            

                    im6 = axes[0,2].imshow(diff_img, cmap='seismic', vmin=-50, vmax=50)
                    axes[0,2].set_title("Difference (Corrected - Ref.)", fontsize=16)
                    axes[0,2].tick_params(axis='both', labelsize=16)

                    # Ajout des cercles verts sur les transients détectés
                    if transients is not None and len(transients) > 0:
                        for source in transients:
                            x, y = source['xcentroid'], source['ycentroid']
                            circ = patches.Circle((x, y), radius=12, edgecolor='g', facecolor='none', lw=2)
                            axes[0,2].add_patch(circ)
                    axes[0,2].legend(loc='upper right', fontsize=16)
                    plt.colorbar(im6, ax=axes[0,2], fraction=0.046, pad=0.04)
                    
                    plt.suptitle(f"Comparison : {os.path.basename(fits_file)}", fontsize=16)
                    plt.tight_layout()
                    plt.show()
                    
        if len(tab_data) == 0:
            print("Aucun transient détecté.")
            return None
        final_table = vstack(tab_data)
        # ========== Groupement spatial des transients ==========
        clusters = self.group_transients_by_position(final_table, tol=3)

        print(f"{len(final_table)} détections totales regroupées en {len(clusters)} transients uniques.")

        # Tu pourrais stocker les clusters pour analyse ultérieure si besoin :
        # self.transient_clusters = clusters

        return final_table
    
    
    
    
    
    
    
    
    def detect_stars(self, image_data, fwhm=3.0, threshold_sigma=5.0):
        """Détecte les étoiles dans une image donnée.
        
        Parameters
        ----------
        image_data : 2D array
            Les données de l'image.
        fwhm : float
            FWHM des étoiles (approximatif).
        threshold_sigma : float
            Seuil de détection en unités de sigma.
        
        Returns
        -------
        list of dict
            Liste des étoiles, chaque élément ayant {'x': float, 'y': float}.
        """
        mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
        daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma * std)
        sources = daofind(image_data - median)

        if sources is None:
            return []
        return [{'x': x, 'y': y} for x, y in zip(sources['xcentroid'], sources['ycentroid'])]
        
        
        
        
        

    def group_positions(self, all_positions, tol=3.0, min_count=2, max_count=None):
        """Regroupe des positions répétées à travers plusieurs images.

        Parameters
        ----------
        all_positions : list of dict
            Liste des positions des étoiles.
        tol : float
            Tolérance en pixels pour regrouper deux positions.
        min_count : int
            Nombre minimal d'observations pour conserver la position.

        Returns
        -------
        list of dict
            Liste des positions groupées avec {'id': int, 'x': float, 'y': float, 'count': int}.
        """
        positions_array = np.array([[pos["x"], pos["y"]] for pos in all_positions])
        used = np.zeros(len(positions_array), dtype=bool)
        grouped_positions = []
        id_counter = 1

        for i in range(len(positions_array)):
            if used[i]:
                continue
            cluster_indices = [i]
            used[i] = True
            for j in range(i + 1, len(positions_array)):
                if not used[j] and np.linalg.norm(positions_array[i] - positions_array[j]) < tol:
                    cluster_indices.append(j)
                    used[j] = True
            if len(cluster_indices) >= min_count:
                if max_count is None or len(cluster_indices) > max_count:
                    avg_x = np.mean(positions_array[cluster_indices, 0])
                    avg_y = np.mean(positions_array[cluster_indices, 1])
                    grouped_positions.append({
                        'id': id_counter,
                        'x': avg_x,
                        'y': avg_y,
                        'count': len(cluster_indices)
                    })
                    id_counter += 1
        return grouped_positions
    
    
    
    
    
    def get_diff_positions(self, aligned_files, ref_filename, date_range, set_zp=None, 
                           limit_seeing=None, threshold_sigma=5.0, cluster_tol=3.0, min_count=5, max_count=None, show_plots=False):
        """Détecte toutes les positions répétées des étoiles observables
        à partir des images de différence.

        Parameters
        ----------
        aligned_files : list
            Liste des chemins des FITS alignés.
        ref_filename : str
            Chemin du FITS de référence.
        date_range : list
            Date de début et de fin au format [[YYYY_start, YYYY_end],
                                            [MM_start, MM_end],
                                            [DD_start, DD_end]]
        fwhm : float
            FWHM utilisé pour la détection des étoiles.
        threshold_sigma : float
            Seuil en sigma pour la détection des étoiles.
        tol : float
            Tolérance en pixels pour regrouper des positions légèrement décalées.
        min_count : int
            Nombre minimal de répétitions nécessaires pour conserver une position.

        Returns
        -------
        list of dict
            Liste des positions répétées avec leur position moyenne:
            [{'x': float, 'y': float, 'count': int}, ...]
        """
        ref_hdu = fits.open(ref_filename)[0]
        image_ref = ref_hdu.data
        image_ref -= np.nanmedian(image_ref)
        zp_ref = ref_hdu.header["MAGZP"]
        seeing_ref = ref_hdu.header["SEEING"]

        (year_start, year_end), (month_start, month_end), (day_start, day_end) = date_range
        start_date = datetime(year_start, month_start, day_start)
        end_date = datetime(year_end, month_end, day_end)
        start_mjd = Time(start_date).mjd
        end_mjd = Time(end_date).mjd

        all_detected_positions = []
        i = 0  
        
        if set_zp is not None:
            image_ref = image_ref * 10 ** ((set_zp - zp_ref)/2.5)
            zp_ref = set_zp
        if limit_seeing is not None:
            seeing_ref = limit_seeing
        
        for fits_file in aligned_files:
            with fits.open(fits_file) as hdu_list:
                hdu = hdu_list[0]
                mjd = hdu.header.get("OBSMJD")
                
                # ---- Filtrage par date ----
                if mjd is None or not (start_mjd <= mjd <= end_mjd):
                    continue

                img_data = hdu.data
                saturate = hdu.header["SATURATE"]
                zp_img = hdu.header["MAGZP"]
                seeing_img = hdu.header["SEEING"]

                # ---- Correction des saturations ----
                img_data[img_data >= saturate] = np.nan

                # ---- Soustraction du fond de ciel ----
                if np.nanmedian(img_data) > 400:
                    img_data_wob, bkg_map = self.remove_local_background(img_data, block_size=64)
                else:
                    img_data_wob = img_data - np.nanmedian(img_data)

                # ---- Dégradation du seeing ----
                img_data_corr = img_data_wob * 10 ** ((zp_ref - zp_img) / 2.5)

                if seeing_img < seeing_ref:
                    img_data_corr = self.degrade_psf(img_data_corr, seeing_img, seeing_ref)
                elif seeing_ref < seeing_img:
                    image_ref = self.degrade_psf(image_ref, seeing_ref, seeing_img)

                diff_img = img_data_corr - image_ref
                diff_img -= np.nanmedian(diff_img)
                
                # ---- Détection des étoiles ----
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=AstropyWarning)
                    stars = self.detect_stars(diff_img, fwhm=seeing_ref, threshold_sigma=threshold_sigma)

                if stars:
                    all_detected_positions.extend(stars)
                #==============================# DATES AND FRACDAY #=============================#
                date_obs = Time(mjd, format='mjd').to_datetime()
                date_display = date_obs.strftime("%Y-%m-%d")
                fracday = f"{date_obs.hour:02}{date_obs.minute:02}{date_obs.second:02}"    
                #================================# Plot #================================#
                if show_plots:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    im = ax.imshow(diff_img, cmap='seismic', vmin=-50, vmax=50)
                    ax.set_title(f"Diff {date_display} | {fracday}", fontsize=16)
                    if stars:
                        for star in stars:
                            circ = patches.Circle((star['x'], star['y']), radius=15, edgecolor='green', facecolor='none', linewidth=2)
                            ax.add_patch(circ)
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    plt.show()
                
                i += 1
                print(f"Image n°{i} processed ({date_display} | {fracday}) , found {len(stars)} stars")
                
        # ---- Regroupement des positions répétées ----
        grouped_positions = self.group_positions(all_detected_positions, tol=cluster_tol, min_count=min_count, max_count=max_count)
        print(f"Total positions détectées : {len(all_detected_positions)}")
        print(f"Positions regroupées : {len(grouped_positions)}")

        return grouped_positions






    def get_data(self, aligned_files, ref_filename, positions, date_range, set_zp=None, aperture=15):
            """Extrait les flux des positions spécifiques à travers les images.

            Parameters
            ----------
            positions : list of dict
                Positions spécifiques à analyser avec des clés {'id', 'x', 'y'}.

            Returns
            -------
            astropy.table.Table
                Table avec les colonnes:
                'date', 'fracday', 'ra', 'dec', 'x', 'y', 'flux', 'mag_r', 'id'
            """
            ref_hdu = fits.open(ref_filename)[0]
            image_ref = ref_hdu.data
            image_ref -= np.nanmedian(image_ref)
            zp_ref = ref_hdu.header['MAGZP']
            seeing_ref = ref_hdu.header['SEEING']
            wcs_ref = WCS(ref_hdu.header)

            (year_start, year_end), (month_start, month_end), (day_start, day_end) = date_range
            start_date = datetime(year_start, month_start, day_start)
            end_date = datetime(year_end, month_end, day_end)
            start_mjd = Time(start_date).mjd
            end_mjd = Time(end_date).mjd
            
            results = []
            
            if set_zp is not None:
                image_ref = image_ref * 10 ** ((set_zp - zp_ref)/2.5)
                zp_ref = set_zp
                
            #==========================================# GET DATA #=======================================#
            for fits_file in aligned_files:
                with fits.open(fits_file) as hdu_list:
                    hdu = hdu_list[0]
                    mjd = hdu.header["OBSMJD"]
                    
                    #=======================# Filtering FITS files by dates #=========================#
                    if mjd is None or not (start_mjd <= mjd <= end_mjd):
                        continue
                    #==============================# DATES AND FRACDAY #=============================#
                    date_obs = Time(mjd, format='mjd').to_datetime()
                    date_display = date_obs.strftime("%Y-%m-%d")
                    fracday = f"{date_obs.hour:02}{date_obs.minute:02}{date_obs.second:02}"
                    #==============================# IMAGE CORRECTION #=============================#
                    
                    img_data = hdu.data
                    saturate = hdu.header['SATURATE']
                    zp_img = hdu.header["MAGZP"]
                    seeing_img = hdu.header["SEEING"]
                    
                    img_data[img_data >= saturate] = np.nan
                    #--------------------------------# Background removal #-------------------------------#
                    if np.nanmedian(img_data) > 400:
                        img_data_wb, bkg_map = self.remove_local_background(img_data, block_size=64)
                    else:
                        img_data_wb = img_data - np.nanmedian(img_data)
                    #--------------------------------# Zero Point Correction #-------------------------------#
                    img_data_corr = img_data_wb * 10**((zp_ref - zp_img)/2.5)
                    #--------------------------------# PSF Degradation #-------------------------------#    
                    if seeing_img < seeing_ref:
                        img_data_corr = self.degrade_psf(img_data_corr, seeing_img, seeing_ref)
                    elif seeing_ref < seeing_img:
                        image_ref = self.degrade_psf(image_ref, seeing_ref, seeing_img)
                        
                    diff_img = img_data_corr - image_ref
                    diff_img -= np.nanmedian(diff_img)
                    #==============================# GET DATA #==============================#
                    
                    
                    for pos in positions:
                        pos["ra"], pos["dec"] = self.pixel_to_radec(pos["x"], pos["y"], wcs_ref)
                        ra, dec = pos["ra"], pos["dec"]
                        x, y = pos["x"], pos["y"]
                        id_source = pos["id"]
                        phot = self.star_photometry(np.round(x), np.round(y), aperture, diff_img, zp_img)
                        results.append({
                            "date": date_display,
                            "fracday": fracday,
                            "ra": ra,
                            "dec": dec,
                            "x": x,
                            "y": y,
                            "flux": phot["flux"][0],
                            "mag": phot["mag"][0],
                            "id": id_source
                        })
            if results:
                names = list(results[0].keys())
                rows = [[res[name] for name in names] for res in results]
                data = Table(rows=rows, names=names)
                return data
            else:
                return Table()
            
    
    
    
    
    
    
    
    

    def compare_flux_with_ztf(self, final_table, apply_mask=True, coef_sigma=2,
                           poly_order=12, zp_title=None, show_ztf=False, ztf_csv_path=None,
                           shared_yaxis=False, show_fit=True):
        """Affiche toutes les sources de final_table (avec colonne 'id') 
        dans un grand subplot (7 colonnes par ligne).

        Parameters
        ----------
        final_table : astropy.table.Table
            Table de données à afficher, avec colonne 'id' pour identifier les sources.
        apply_mask : bool
            Appliquer le masquage des flux aberrants.
        coef_sigma : float
            Coefficient de sigma utilisé pour le masquage.
        poly_order : int
            Ordre du polynôme utilisé pour le fit.
        zp : float
            Zero point à afficher en légende.
        show_ztf : bool
            Afficher les données ZTF si True.
        ztf_csv_path : str
            Chemin du fichier ZTF à charger.
        shared_yaxis : bool
            Partage de l'axe Y entre Astrotools et ZTF.
        show_fit : bool
            Tracer le fit polynomial.
        """
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        from astropy.time import Time
        import matplotlib.pyplot as plt
        import math

        ids = np.unique(final_table["id"].astype(int))
        n_ids = len(ids)

        # Disposition du subplot
        ncols = 7
        nrows = math.ceil(n_ids / ncols)

        fig, axs = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)

        for idx, source_id in enumerate(ids):
            ax = axs[idx // ncols, idx % ncols]
            source_data = final_table[final_table["id"] == source_id]

            # ---- Masquage Astrotools ----
            med_flux = np.median(source_data["flux"])
            std_flux = np.std(source_data["flux"])
            mask = (source_data["flux"] >= med_flux - coef_sigma * std_flux) & \
                (source_data["flux"] <= med_flux + coef_sigma * std_flux)
            filtered = source_data[mask] if apply_mask else source_data

            if len(filtered) == 0:
                ax.set_title(f"ID {source_id} (no valid data)", fontsize=8)
                ax.axis("off")
                continue

            # ---- Temps Astrotools ----
            filtered["datetime"] = [
                datetime.strptime(row["date"], "%Y-%m-%d") + timedelta(seconds=int(row["fracday"]))
                for row in filtered
            ]
            t0_ast = filtered["datetime"][0]
            t_ast = [(dt - t0_ast).total_seconds() for dt in filtered["datetime"]]

            flux_ast_fit = None
            dt_ast_fit = None
            if show_fit and len(filtered) > poly_order:
                try:
                    coeff_ast = np.polyfit(t_ast, filtered["flux"], poly_order)
                    flux_ast_fit = np.poly1d(coeff_ast)
                    t_fit = np.linspace(min(t_ast), max(t_ast), 500)
                    dt_ast_fit = [t0_ast + timedelta(seconds=s) for s in t_fit]
                except np.linalg.LinAlgError:
                    flux_ast_fit = None
                    dt_ast_fit = None

            # ---- Données ZTF ----
            df = None
            flux_fit_ztf = None
            dt_fit_ztf = None
            if show_ztf:
                df = pd.read_csv(ztf_csv_path)
                df = df[(df["filter"] == "ztfr") & (df["mjd"] >= 58849) & (df["mjd"] <= 58970)]
                if len(df) > 0:
                    df["date"] = Time(df["mjd"], format="mjd").to_datetime()
                    t0_ztf = df["date"].iloc[0]
                    t_ztf = [(d - t0_ztf).total_seconds() for d in df["date"]]
                    if show_fit and len(df) > poly_order:
                        try:
                            coeff_ztf = np.polyfit(t_ztf, df["flux"], poly_order)
                            flux_fit_ztf = np.poly1d(coeff_ztf)
                            t_fit_ztf = np.linspace(min(t_ztf), max(t_ztf), 500)
                            dt_fit_ztf = [t0_ztf + timedelta(seconds=s) for s in t_fit_ztf]
                        except np.linalg.LinAlgError:
                            flux_fit_ztf = None
                            dt_fit_ztf = None

            # ---- Tracé Astrotools ----
            ax.scatter(filtered["datetime"], filtered["flux"], s=30, color="mediumblue", label=f"Astr ({zp_title})")
            if flux_ast_fit is not None:
                ax.plot(dt_ast_fit, flux_ast_fit(t_fit), color="mediumblue", linestyle="--", linewidth=1)
            ax.tick_params(axis="y", labelsize=7, labelcolor="blue")
            ax.legend(fontsize=7, loc="best")

            # ---- Tracé ZTF ----
            if show_ztf and df is not None and len(df) > 0:
                if shared_yaxis:
                    ax.scatter(df["date"], df["flux"], s=30, color="orange", alpha=0.5, label="ZTF (zp=30)")
                    if flux_fit_ztf is not None:
                        ax.plot(dt_fit_ztf, flux_fit_ztf(t_fit_ztf), color="darkorange", linestyle="--", linewidth=1)
                    ax.legend(fontsize=7, loc="best")
                else:
                    ax2 = ax.twinx()
                    ax2.scatter(df["date"], df["flux"], s=30, color="orange", alpha=0.5, label="ZTF (zp=30)")
                    if flux_fit_ztf is not None:
                        ax2.plot(dt_fit_ztf, flux_fit_ztf(t_fit_ztf), color="darkorange", linestyle="--", linewidth=1)
                    ax2.legend(fontsize=7, loc="best")
                    # COLORATION DE L'AXE ZTF
                    ax2.tick_params(axis="y", labelsize=7, labelcolor="orange")

            # ---- Construction du titre ----
            x_source = np.round(np.mean(filtered["x"]), 1)    # Position X moyenne
            y_source = np.round(np.mean(filtered["y"]), 1)    # Position Y moyenne
            ra_source = np.round(np.mean(filtered["ra"]), 5)  # RA moyenne
            dec_source = np.round(np.mean(filtered["dec"]), 5) # Dec moyenne

            title = (f"ID {source_id} | X:{x_source} Y:{y_source}\n"
                    f"RA:{ra_source} Dec:{dec_source}")

            ax.set_title(title, fontsize=8)
            ax.tick_params(axis="both", labelsize=7)

        # ---- Suppression des axes vides ----
        for j in range(n_ids, nrows * ncols):
            axs[j // ncols, j % ncols].set_visible(False)

        # ---- Finalisation ----
        fig.autofmt_xdate()
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()








    def compare_all_sources(self, final_table, apply_mask=True, coef_sigma=2,
                            shared_yaxis=False, show_fit=False, poly_order=12,
                            zp=None, show_ztf=False, ztf_csv_path=None):
        """Trace toutes les courbes de lumière des sources identifiées."""
        ids = np.unique(np.asarray(final_table["id"]))

        for source_id in ids:
            source_data = final_table[final_table["id"] == source_id]
            self.compare_flux_with_ztf(source_data,
                                        apply_mask=apply_mask,
                                        coef_sigma=coef_sigma,
                                        shared_yaxis=shared_yaxis,
                                        show_fit=show_fit,
                                        poly_order=poly_order,
                                        zp=zp,
                                        show_ztf=show_ztf,
                                        ztf_csv_path=ztf_csv_path)
            
            
            
            
        









import csv
import os
import requests
from tqdm import tqdm


class ZTFDownloader:
    def __init__(self, data_type="sciimg" ,output_base_dir='ztf_data'):
        self.base_url = 'https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci/'
        self.data_type = data_type
        self.output_base_dir = output_base_dir
        os.makedirs(self.output_base_dir, exist_ok=True)

    def get_one_file(self, year, month, day, fractime, field, filter_band, ccd, quadrant):
        imgtypecode = 'o'
        zeros = '00'
        datefull = f"{year}{month}{day}{fractime}"
        prefixe = f"ztf_{datefull}_{zeros}{field}_{filter_band}_c{ccd}_{imgtypecode}_q"
        suffixe = f'_{self.data_type}.fits'
        filename = f"{prefixe}{quadrant}{suffixe}"

        output_dir = os.path.join(self.output_base_dir, f"{zeros}{field}_{filter_band}_c{ccd}_{imgtypecode}_q{quadrant}")
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)

        url = f"{self.base_url}/{year}/{month}{day}/{fractime}/{filename}"
        print(f"Téléchargement : {url}")

        response = requests.get(url, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            with open(filepath, 'wb') as f:
                for data in tqdm(response.iter_content(1024), total=total_size // 1024, unit='KB'):
                    f.write(data)
            print(f"Fichier téléchargé : {filepath}")
        elif response.status_code == 404:
            print("Image non trouvée pour les paramètres donnés.")
        else:
            print(f"Erreur de téléchargement : {response.status_code}")

    def download_from_csv(self, csv_path, target_field, filter_band, ccd, quadrant, start_date, end_date):
        output_dir = os.path.join(self.output_base_dir, f"00{target_field}_{filter_band}_c{ccd}_o_q{quadrant}")
        os.makedirs(output_dir, exist_ok=True)

        with open(csv_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            if target_field not in reader.fieldnames:
                print(f"Le champ '{target_field}' n'existe pas dans le CSV.")
                return

            for row in reader:
                filename = row[target_field]
                if filename:
                    date_part = filename[:8]
                    if start_date <= date_part <= end_date:
                        year = filename[:4]
                        month = filename[4:6]
                        day = filename[6:8]
                        fractime = filename[8:]
                        self.get_one_file(year, month, day, fractime, target_field, filter_band, ccd, quadrant)
