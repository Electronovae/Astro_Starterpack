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
from datetime import datetime
from astropy.table import vstack
from matplotlib.patches import Rectangle
from astropy.coordinates import SkyCoord
import glob

class AstroTools:
    
    def __init__(self, fits_files, file_name ='00Field_Filter_cCcd_o_qQuadrant'):
        self.fits_files = fits_files 
        self.first_file = fits_files[0]
        self.first_hdu = fits.open(self.first_file)[0]
        self.first_wcs = WCS(self.first_hdu.header)
        self.first_shape = self.first_hdu.data.shape
        self.file_name = file_name
        
        

    def show_img(self, data, title='', vmin=None, vmax=None, cmap=None):
        plt.figure(figsize=(10, 10))
        plt.imshow(data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title(title)
        plt.show()
    
    
    
    def radec_to_pixel(self, ra, dec, wcs):
        skycoord = SkyCoord(ra, dec, unit='deg')
        x, y = wcs.world_to_pixel(skycoord)
        return x, y



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
            plt.figure(figsize=(10, 6))
            plt.hist(seeing_values, bins=50, color='darkblue', edgecolor = 'orange')
            plt.plot(x_vals, gaussian_curve, color='orange', linewidth=2, label='Gaussian distribution')
            plt.axvline(mean_seeing, color = 'r', label = f'mean seeing = {mean_seeing:.3f}')
            plt.axvline(median_seeing, color = 'g', label = f'median seeing = {median_seeing:.3f}')
            plt.axvline(mean_seeing+sigma_seeing, color = 'r',  linestyle = '--')
            plt.axvline(mean_seeing-sigma_seeing, color = 'r',  linestyle = '--', label = f'sigma seeing = {sigma_seeing:.3f}')
            plt.title(f"seeing distribution ({self.file_name})")
            plt.xlabel("Seeing")
            plt.ylabel("Nomber of images")
            plt.tight_layout()
            plt.legend()
            plt.show()
        return mean_seeing, median_seeing , sigma_seeing , seeing_max 



    def align_images(self,output_dir=None,show=False):  # ~12 seconds/image
        if output_dir is None:
            output_dir = f"ztf_data_aligned\\{self.file_name}\\"
        else:
            output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        numero = 0
        
        for fits_file in self.fits_files:
            with fits.open(fits_file) as hdu_list:
                hdu = hdu_list[0]
                source_wcs = WCS(hdu.header)
                filename = os.path.basename(fits_file)

                # Reprojection
                reprojected_data = self.reproject_astropy(hdu.data, source_wcs, self.first_wcs, self.first_shape)

                # Update header
                new_header = self.first_hdu.header.copy()
                for key in ['SEEING', 'SATURATE', 'MAGZP']:
                    if key in hdu.header:
                        new_header[key] = hdu.header[key]
                new_header['NAXIS'] = 2
                new_header['NAXIS1'] = reprojected_data.shape[1]
                new_header['NAXIS2'] = reprojected_data.shape[0]

                # Save file
                aligned_filename = os.path.join(output_dir, f"aligned_{filename}")
                new_hdu = fits.PrimaryHDU(data=np.array(reprojected_data.astype(np.float32)), header=new_header)
                new_hdu.writeto(aligned_filename, overwrite=True)
                numero += 1

                print(f"Image aligned saved : {aligned_filename}")
                print(f'Image n° {numero}')

                if show:
                    vmed = np.nanmedian(reprojected_data)
                    self.show_img(reprojected_data, f"Aligned {filename}", vmin=vmed, vmax=vmed+100)

        print("Alignment completed for all valid images !")
        
        
        
    def get_refimg(self, aligned_files, ref_method='median', seeing_max=6, zp_ref=None, limit_bkg=400, Imshow_ref=True, Imshow=False, output_file='ztf_data_ref', n=''): 

        num_images = 0
        first_hdu_a = fits.open(aligned_files[0])[0]
        valid_paths = []
        
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
                sky_bkg = np.nanmedian(img_data)
                
                if sky_bkg > limit_bkg:
                    print(f"Image ignored because background too bright (clouds) : {fits_file}, median = {sky_bkg}")
                    continue
                if seeing > seeing_max:
                    print(f"Image ignored due to excessive seeing : {fits_file}, seeing = {seeing}")
                    continue
                
                num_images += 1
                img_data[img_data >= saturate] = np.nan # Skip (into nan) saturate values
                img_data = img_data - sky_bkg    
                img_data_corr = img_data * 10**((zp_ref - zp_img)/2.5)
                img_data_corr = self.degrade_psf(img_data_corr, seeing, seeing_max, pixel_scale=1.01)
                print('mag = ', zp_img)
                print ('seeing = ', seeing )
                print(f'n°{num_images}')
                
                if ref_method == 'median':
                    temp_path = os.path.join(temp_dir, f"img_{num_images:04d}.npy")
                    np.save(temp_path, img_data_corr.astype(np.float32))
                    valid_paths.append(temp_path)
                if ref_method == 'mean':
                    image_sum += np.array(img_data_corr, dtype=np.float32)
            
                
                #========================================# PLOTS #========================================#
                if Imshow == True:
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    im = axes[0].imshow(img_data_corr, vmin = 0, vmax = 100)
                    axes[0].set_title(f"Corrected image : {os.path.basename(fits_file)}")
                    plt.colorbar(im, ax=axes[0], orientation='vertical')
                    
                    axes[1].hist(img_data.flatten(), bins=100, alpha=0.5, color = 'b', label='Before correction')
                    axes[1].hist(img_data_corr.flatten(), bins=100, alpha=0.5, color = 'r', label='After correction')
                    axes[1].legend()
                    axes[1].set_title("Flux Histogram")
                    axes[1].set_xlabel("Flux")
                    axes[1].set_ylabel("Number of pixels")
                    axes[1].set_yscale('log')
                    
                    im0 = axes[2].imshow(img_data_corr[2000:2500,1500:2000], vmin=0, vmax=100)
                    axes[2].set_title(f"Corrected image : {os.path.basename(fits_file)}")
                    plt.colorbar(im0, ax=axes[2])
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

            basename = os.path.basename(aligned_files[0])
            date_str = basename.split('_')[2][:8]
            fracday = basename.split('_')[2][8:14]

            output_filename = os.path.join(output_path, f"refimg_median_{date_str}{fracday}_{self.file_name}_{n}.fits")

            new_hdu = fits.PrimaryHDU(data=image_median.astype(np.float32), header=first_hdu_a.header)
            new_hdu.header['SEEING'] = (round(float(seeing_max), 3), 'Mean seeing used to build image_median')
            new_hdu.header['MAGZP'] = zp_ref
            new_hdu.header['NFRAMES'] = (num_images, 'Number of images used for image_median')
            new_hdu.writeto(output_filename, overwrite=True)
            print(num_images, 'Number of images used for image_median')
            
            if Imshow_ref == True:
                self.show_img(data=image_median, title="Reference Image (Median)", vmin = 0, vmax = 100)
            print(f"Recorded median reference image : {output_filename}")
            
            del rows
            shutil.rmtree(temp_dir)
        #========================================# Mean #========================================#
        if ref_method == 'mean':
            image_avg = image_sum / num_images
            output_path = f"{output_file}\\{self.file_name}\\"
            os.makedirs(output_path, exist_ok=True)
            basename = os.path.basename(aligned_files[0])
            date_str = basename.split('_')[2][:8]
            fracday = basename.split('_')[2][8:14]
            output_filename = os.path.join(output_path, f"refimg_mean_{date_str}{fracday}_{self.file_name}_{n}.fits")
            new_hdu = fits.PrimaryHDU(data=np.array(image_avg), header=first_hdu_a.header)
            new_hdu.header['SEEING'] = (round(float(seeing_max), 3), 'Mean seeing used to build image_avg')
            new_hdu.header['NFRAMES'] = (num_images, 'Number of images used for image_avg')
            new_hdu.writeto(output_filename, overwrite=True)
            print(num_images, 'Number of images used for image_avg')
            
            if Imshow_ref == True:
                self.show_img(data=image_avg, title="Reference Image (Mean)", vmin = 0, vmax = 100)
            print(f"Recorded mean reference image : {output_filename}")
            
        
        
    def subtraction(self, aligned_files , ref_filename , ra , dec , Start_Year, 
                    Start_Month, Start_Day , End_Year, End_Month, End_Day, zp=None, limit_bkg=500, Imshow=True, zoom=50 , coef_img_ref=1): #seeing_max est donné par la fonction distrib_seeing()
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
        fits_file_aligned_date_filtered = []
        dates = []
        fracday_list = []
        tab_data = []
        
        if zp is not None:
            zp_ref = zp

    #=============================# Filtering FITS files by dates #=============================#
        for file in aligned_files:
            basename = os.path.basename(file)
            date_str = basename.split('_')[2][:8]  # ex: '20200115'
            try:
                file_date = datetime.strptime(date_str, '%Y%m%d')
                if start_date <= file_date <= end_date:
                    fits_file_aligned_date_filtered.append(file)
            except ValueError:
                continue
        
        for fits_file in fits_file_aligned_date_filtered:
            with fits.open(fits_file) as hdu_list:
                hdu = hdu_list[0]
                img_data = hdu.data
                wcs = WCS(hdu.header)
                zp_img = hdu.header['MAGZP']
                seeing_img = hdu.header['SEEING']
                saturate = hdu.header['SATURATE']
                
                if np.nanmedian(img_data) > limit_bkg:
                    print(f"Image ignored for background too bright (clouds) : {fits_file}, median = {np.nanmedian(img_data)}")
                    continue
                if seeing_img > seeing_ref:
                    print(f"Image ignored for seeing out of range : {fits_file}, seeing = {seeing_img}")
                    continue
                #==============================# DATES AND FRACDAY #=============================#
                basename = os.path.basename(fits_file)
                date_str = basename.split('_')[2][:8]
                fracday = basename.split('_')[2][8:14]
                file_date = datetime.strptime(date_str, '%Y%m%d')
                date_display = file_date.strftime('%Y-%m-%d')
                dates.append(date_display)
                fracday_list.append(fracday)
                #==============================# IMAGE CORRECTION #=============================#
                img_data[img_data >= saturate] = np.nan
                img_data = img_data - np.nanmedian(img_data)
                img_data_corr = img_data * 10**((zp_ref - zp_img)/2.5)
                
                if seeing_img < seeing_ref:
                    img_data_corr = self.degrade_psf(img_data_corr, seeing_img, seeing_ref)
                elif seeing_ref < seeing_img:
                    image_ref = self.degrade_psf(image_ref, seeing_ref, seeing_img)
                    
                num_images += 1
                image_ref *= coef_img_ref
                diff_img = img_data_corr - image_ref
                x_min , x_max , y_min , y_max = self.get_zoom(img_data_corr, ra , dec , wcs , zoom=zoom)
                x_pixel, y_pixel = self.radec_to_pixel(ra, dec, wcs)
                #==============================# PHOTOMETRY #=============================#
                ap = aphoto_modified(hdu_list)
                photometry = ap.photometry([(x_pixel, y_pixel, 15)])
                photometry['flux_r'] = photometry['flux_r'].astype(np.float32)
                photometry['flux_r'] *= 10**((zp_ref - zp_img)/2.5)
                tab_data.append(photometry)
                print ('SEEING = ', seeing_img )
                print ('Image n° ', num_images )
                print(date_display)
                
                if Imshow == True:
                    #==============================# PLOTS #=============================#
                    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
                    #-------------------#First Line#-------------------#
                    im0 = axes[0,0].imshow(img_data_corr[y_min:y_max, x_min:x_max], vmin=0, vmax=100) 
                    axes[0,0].set_title(f"{date_display} | fractime : {fracday} (zoom)")
                    plt.colorbar(im0, ax=axes[0,0], fraction=0.046, pad=0.04)
                    im1 = axes[0,1].imshow(image_ref[y_min_ref:y_max_ref, x_min_ref:x_max_ref], vmin=0, vmax=100)
                    axes[0,1].set_title("Ref img (zoom)")
                    plt.colorbar(im1, ax=axes[0,1], fraction=0.046, pad=0.04)           
                    im2 = axes[0, 2].imshow(img_data_corr, vmin=0, vmax=10)
                    axes[0,2].set_title(f"{date_display} | fractime : {fracday}")
                    rect1 = Rectangle((x_min_ref, y_min_ref), x_max_ref - x_min_ref, y_max_ref - y_min_ref, linewidth=1, edgecolor='r', facecolor='none', label='Zoom (ref)')
                    rect2 = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='orange', facecolor='none', label='Zoom')
                    axes[0,2].add_patch(rect1)
                    axes[0,2].add_patch(rect2)
                    axes[0, 2].legend(loc='upper right')
                    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
                    im3 = axes[0,3].imshow(image_ref, vmin=0, vmax=10)
                    axes[0,3].set_title("Ref img")
                    plt.colorbar(im3, ax=axes[0,3], fraction=0.046, pad=0.04)           
                    #-------------------#Second Line#-------------------#
                    im4 = axes[1,0].imshow(diff_img[y_min_ref:y_max_ref, x_min_ref:x_max_ref], cmap='seismic', vmin=-50, vmax=50)
                    axes[1,0].set_title("Difference (Corrected - Ref.) (zoom)")
                    plt.colorbar(im4, ax=axes[1,0], fraction=0.046, pad=0.04)            
                    im5 = axes[1,1].imshow(np.clip(diff_img, 0, None)[y_min_ref:y_max_ref, x_min_ref:x_max_ref], cmap='seismic', vmin=-50, vmax=50)
                    axes[1,1].set_title("Difference (Corrected - Ref.) without negative values (zoom)")
                    plt.colorbar(im5, ax=axes[1,1], fraction=0.046, pad=0.04)
                    im6 = axes[1,2].imshow(diff_img, cmap='seismic', vmin=-50, vmax=50)
                    axes[1,2].set_title("Difference (Corrected - Ref.)")
                    plt.colorbar(im6, ax=axes[1,2], fraction=0.046, pad=0.04)
                    im7 = axes[1,3].imshow(np.clip(diff_img, 0, None), cmap='seismic', vmin=-50, vmax=50)
                    axes[1,3].set_title("Difference (Corrected - Ref.) without negative values")
                    plt.colorbar(im7, ax=axes[1,3], fraction=0.046, pad=0.04)
                    plt.suptitle(f"Comparison : {os.path.basename(fits_file)}", fontsize=14)
                    plt.tight_layout()
                    plt.show()  
        final_table = vstack(tab_data)
        final_table['date'] = dates
        final_table['fracday'] = fracday_list
        print(final_table)
        return final_table