import napari
import torch
from skimage.measure import regionprops_table, regionprops
from skimage.color import label2rgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cellpose
from cellpose import models
import edt
import glob
import os
import tqdm
from matplotlib_scalebar.scalebar import ScaleBar
import gc
import seaborn as sns
from sklearn.mixture import GaussianMixture
from bioio import BioImage
import bioio_nd2
import bioio_tifffile
from bioio.writers import OmeTiffWriter
import scipy.stats as stats
from scipy import optimize
from scipy.ndimage import map_coordinates #new
import bioio

nucChannel = 0 # red emerin rings
spotChannel = 1 # green spots

# # to include in the actuall script
# raw_input_path = '/mnt/external.data/MeisterLab/Dario/SDC1/1273/20240813_e/'
# output_path = '/mnt/external.data/MeisterLab/mvolosko/image_project/sdc1/'+protein_strain_date+'/'
#denoised_input_path = raw_input_path+'N2V_sdc1_dpy27_mSG_emr1_mCh/denoised/'

protein_strain_date = '/'.join(os.path.normpath(raw_input_path).split(os.sep)[-3:])


if not os.path.exists(output_path+"/qc"):
    os.makedirs(output_path+"/qc")

if not os.path.exists(output_path+"/segmentation"):
    os.makedirs(output_path+"/segmentation")

if not os.path.exists(output_path+"/edt"):
    os.makedirs(output_path+"/edt")

if not os.path.exists(output_path+"/spots"):
    os.makedirs(output_path+"/spots")

if not os.path.exists(output_path+"/nuclei"):
    os.makedirs(output_path+"/nuclei")
    
if not os.path.exists(output_path+"/dist"):
    os.makedirs(output_path+"/dist")


raw_file_name_pattern = "/*.nd2"
denoised_file_name_pattern = "/*_n2v.tif"
raw_filepaths = sorted(glob.glob(raw_input_path + raw_file_name_pattern,recursive=True))
raw_filepaths = [filepath for filepath in raw_filepaths if '_bad.nd2' not in filepath]

print(f"Found {len(raw_filepaths)} *.nd2 files.")



# Generate data frame of file path with metadata
df = pd.DataFrame()
df['filename'] = [os.path.basename(filepath)[:-4] for filepath in raw_filepaths]
tmpdate = [os.path.normpath(filepath).split(os.sep)[-2] for filepath in raw_filepaths]
df['date'] = pd.Series([exp.split('_')[0] for exp in tmpdate])
df['experiment'] = pd.Series([exp.split('_')[1] for exp in tmpdate])
df['strain'] = [os.path.normpath(filepath).split(os.sep)[-3] for filepath in raw_filepaths]
df['protein'] = [os.path.normpath(filepath).split(os.sep)[-4] for filepath in raw_filepaths]
df['id'] = df['protein'] + '_' + df['experiment'] + '_' + df['filename'] 
df['raw_filepath'] = raw_filepaths
df['denoised_filepath'] = [denoised_input_path+filename+'_n2v.tif' for filename in df['filename']]
df.to_csv(output_path+'fileList.csv',index=False)


# Load model
torch.cuda.device(0)
model_path='/mnt/external.data/MeisterLab/lhinder/segmentation_3d_anja/code/worms_1000epochs_v0'

if torch.cuda.is_available():
    print("GPU is available")
    model = models.CellposeModel(pretrained_model=model_path, gpu=True, device =torch.device('cuda:0'))
else:
    print("Only CPU is available")
    model = models.CellposeModel(pretrained_model=model_path, gpu=False)



# Functions

def segment_nuclei(img, model):
    ''' use pytorch cellpose model to segment nuclei'''
    masks,flows,styles = model.eval(img,do_3D=False,stitch_threshold=0.3,cellprob_threshold =0,diameter =36)
    return masks,flows,styles

def calc_distance_mask(masks,anisotropy):
    '''Calculate the distance map from the nuclei-edge towards the center of nucleus'''
    masks_edt = edt.edt(masks,anisotropy = anisotropy)
    return masks_edt


def plot_qc_nuclei_crop(df, index, df_region_props, img, t=0, display = False, seed=1):
    '''Plot a cropped region of a random sample of 10 nuclei from each image'''
    nb_nuc = 10
    np.random.seed(seed)
    indices_to_sample = np.random.choice(range(len(df_region_props)),size = nb_nuc,replace = False)
    # sort indeces in descending order of area

    widths=[df_region_props['image'][i].shape[1] for i in indices_to_sample]

    fig, axs = plt.subplots(nrows = 2, ncols = nb_nuc, figsize = (15,5),dpi = 250, 
                            sharex=False, sharey=False, width_ratios=widths)
    fig.suptitle(f'Cropped nuclei {df.id.iloc[index]}', fontsize=16)

    for i,sample in enumerate(indices_to_sample):
        intensity_image = df_region_props['intensity_image'][sample][:,:,:,spotChannel] #show first spot channel
        image = df_region_props['image'][sample]
        mx = np.ma.masked_array(intensity_image,mask = ~image)
        z_height = image.shape[0] 
        axs[0,i].imshow(mx[int(z_height/2)])
        axs[0,i].spines['top'].set_visible(False)
        axs[0,i].spines['right'].set_visible(False)
        axs[0,i].spines['bottom'].set_visible(False)
        axs[0,i].spines['left'].set_visible(False)
        axs[0,i].get_xaxis().set_ticks([])
        axs[0,i].get_yaxis().set_ticks([])

    for i,sample in enumerate(indices_to_sample):
        intensity_image = df_region_props['intensity_image'][sample][:,:,:,nucChannel] #show second nuclear channel
        image = df_region_props['image'][sample]
        mx = np.ma.masked_array(intensity_image,mask = ~image)
        z_height = image.shape[0]
        axs[1,i].imshow(mx[int(z_height/2)])
        axs[1,i].spines['top'].set_visible(False)
        axs[1,i].spines['right'].set_visible(False)
        axs[1,i].spines['bottom'].set_visible(False)
        axs[1,i].spines['left'].set_visible(False)
        axs[1,i].get_xaxis().set_ticks([])
        axs[1,i].get_yaxis().set_ticks([])

        if i == nb_nuc-1:
            scalebar = ScaleBar(0.065, "um", length_fraction=1, box_alpha=0.7,color='black',location='lower right',height_fraction = 0.05,border_pad =-1)
            axs[1,i].add_artist(scalebar)

    #plt.tight_layout()
    fig.savefig(output_path + 'qc/cropped_nuclei_'+df.id.iloc[index]+'_t'+'{:02d}'.format(t)+'.pdf')
    if display == False:
        plt.close()
    else:
        plt.show()


def plot_single_nucleus_crop(df, index, df_region_props, nuc_index, img):
    '''Plot a cropped region of a particular nucleus'''

    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (3,1.5),dpi = 250, sharey=True)
    fig.suptitle(f'{df.id.iloc[index]}', fontsize=6)

    intensity_image = df_region_props['intensity_image'][nuc_index][:,:,:,spotChannel] #show first spot channel
    image = df_region_props['image'][nuc_index]
    mx = np.ma.masked_array(intensity_image, mask = ~image)
    z_height = image.shape[0] 
    axs[0].imshow(mx[int(z_height/2)])
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['bottom'].set_visible(False)
    axs[0].spines['left'].set_visible(False)
    axs[0].get_xaxis().set_ticks([])
    axs[0].get_yaxis().set_ticks([])


    intensity_image = df_region_props['intensity_image'][nuc_index][:,:,:,nucChannel] #show second nuclear channel
    image = df_region_props['image'][nuc_index]
    mx = np.ma.masked_array(intensity_image, mask = ~image)
    z_height = image.shape[0]
    axs[1].imshow(mx[int(z_height/2)])
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['bottom'].set_visible(False)
    axs[1].spines['left'].set_visible(False)
    axs[1].get_xaxis().set_ticks([])
    axs[1].get_yaxis().set_ticks([])


    scalebar = ScaleBar(0.065, "um", length_fraction=1, box_alpha=0.7,color='black',location='lower right',height_fraction = 0.05,border_pad =-1)
    axs[1].add_artist(scalebar)

    plt.show()


def plot_qc_segmentation_xyz(img, masks, index, df, t=0, display_plot=False, plotContours=False):
    '''Plot a 2x3 grid of xy, xz, yz slices of the image and the corresponding segmentation'''
    nucChannel = 0
    num_z=img.shape[1]
    num_y=img.shape[2]
    num_x=img.shape[3]
    nlabel=100

    fig = plt.figure(layout='constrained',dpi=450,figsize = (10,10))
    fig.suptitle(f'Segmentation for {df.id.iloc[index]}', fontsize=10)
    subfigs = fig.subfigures(2, 1, wspace=0.1)

    axsTop = subfigs[0].subplots(2, 3,sharex=True, sharey=True)
    #xy
    axsTop[0,0].imshow(label2rgb(masks[int(num_z*0.3),:,:],bg_label=0,bg_color=(255, 255, 255),colors=np.random.random((nlabel, 3))))
    axsTop[1,0].set_title('z='+str(int(num_z*0.3)), fontsize=8)
    axsTop[0,1].imshow(label2rgb(masks[int(num_z*0.5),:,:],bg_label=0,bg_color=(255, 255, 255),colors=np.random.random((nlabel, 3))))
    axsTop[1,1].set_title('z='+str(int(num_z*0.5)), fontsize=8)
    axsTop[0,2].imshow(label2rgb(masks[int(num_z*0.7),:,:],bg_label=0,bg_color=(255, 255, 255),colors=np.random.random((nlabel, 3))))
    axsTop[1,2].set_title('z='+str(int(num_z*0.7)), fontsize=8)

    axsTop[1,0].imshow(img[nucChannel,int(num_z*0.3),:,:],cmap = 'gray_r')
    axsTop[1,1].imshow(img[nucChannel,int(num_z*0.5),:,:],cmap = 'gray_r')
    axsTop[1,2].imshow(img[nucChannel,int(num_z*0.7),:,:],cmap = 'gray_r')

    if plotContours:
        axsTop[1,0].contour(masks[int(num_z*0.3),:,:], [0.5], linewidths=0.5, colors='r')
        axsTop[1,1].contour(masks[int(num_z*0.5),:,:], [0.5], linewidths=0.5, colors='r')
        axsTop[1,2].contour(masks[int(num_z*0.7),:,:], [0.5], linewidths=0.5, colors='r')



    for axss in axsTop:
        for ax in axss:
            #ax.set_xlim(0,num_x)
            #ax.set_ylim(0,num_y)
            ax.set_xticks([])
            ax.set_yticks([])

    axsBottom = subfigs[1].subplots(4, 3,sharex=True,sharey=True)
    #xz
    axsBottom[0,0].imshow(label2rgb(masks[:,int(num_y*0.3),:],bg_label=0,bg_color=(255, 255, 255),colors=np.random.random((nlabel, 3))))
    axsBottom[1,0].set_title('y='+str(int(num_y*0.3)), fontsize=8)
    axsBottom[0,1].imshow(label2rgb(masks[:,int(num_y*0.5),:],bg_label=0,bg_color=(255, 255, 255),colors=np.random.random((nlabel, 3))))
    axsBottom[1,1].set_title('y='+str(int(num_y*0.5)), fontsize=8)
    axsBottom[0,2].imshow(label2rgb(masks[:,int(num_y*0.7),:],bg_label=0,bg_color=(255, 255, 255),colors=np.random.random((nlabel, 3))))
    axsBottom[1,2].set_title('y='+str(int(num_y*0.7)), fontsize=8)

    axsBottom[1,0].imshow(img[nucChannel,:,int(num_y*0.3),:],cmap = 'gray_r')
    axsBottom[1,1].imshow(img[nucChannel,:,int(num_y*0.5),:],cmap = 'gray_r')
    axsBottom[1,2].imshow(img[nucChannel,:,int(num_y*0.7),:],cmap = 'gray_r')

    if plotContours:
        axsBottom[1,0].contour(masks[:,int(num_y*0.3),:], [0.5], linewidths=0.5, colors='r')
        axsBottom[1,1].contour(masks[:,int(num_y*0.5),:], [0.5], linewidths=0.5, colors='r')
        axsBottom[1,2].contour(masks[:,int(num_y*0.7),:], [0.5], linewidths=0.5, colors='r')


    #yz
    axsBottom[2,0].imshow(label2rgb(masks[:,:,int(num_x*0.3)],bg_label=0,bg_color=(255, 255, 255),colors=np.random.random((nlabel, 3))))
    axsBottom[3,0].set_title('x='+str(int(num_x*0.3)), fontsize=8)
    axsBottom[2,1].imshow(label2rgb(masks[:,:,int(num_x*0.5)],bg_label=0,bg_color=(255, 255, 255),colors=np.random.random((nlabel, 3))))
    axsBottom[3,1].set_title('x='+str(int(num_x*0.5)), fontsize=8)
    axsBottom[2,2].imshow(label2rgb(masks[:,:,int(num_x*0.7)],bg_label=0,bg_color=(255, 255, 255),colors=np.random.random((nlabel, 3))))
    axsBottom[3,2].set_title('x='+str(int(num_x*0.7)), fontsize=8)

    axsBottom[3,0].imshow(img[nucChannel,:,:,int(num_x*0.3)],cmap = 'gray_r')
    axsBottom[3,1].imshow(img[nucChannel,:,:,int(num_x*0.5)],cmap = 'gray_r')
    axsBottom[3,2].imshow(img[nucChannel,:,:,int(num_x*0.7)],cmap = 'gray_r')

    if plotContours:
        axsBottom[3,0].contour(masks[:,:,int(num_x*0.3)], [0.5], linewidths=0.5, colors='r')
        axsBottom[3,1].contour(masks[:,:,int(num_x*0.5)], [0.5], linewidths=0.5, colors='r')
        axsBottom[3,2].contour(masks[:,:,int(num_x*0.7)], [0.5], linewidths=0.5, colors='r')

    for axss in axsBottom:
        for ax in axss:
            #ax.set_ylim(0,num_z)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    if display_plot:
        plt.show()
    else:
        fig.savefig(output_path + 'qc/segmentation_'+df.id.iloc[index]+'_t'+'{:02d}'.format(t)+'.png')
        plt.close()


# Run the segmentation script on all images (reserve more than 24GB!)
#this produces segmentation, segmentation_qc and edt files
def run_nuclear_segmentation(indices, df, rerun=False):
    '''Run the segmentation on all images in the dataframe'''
    for index in tqdm.tqdm(indices):
        if rerun or not os.path.exists(output_path+'edt/'+df.id.iloc[index]+'_t0.tif'):
            # get anisotropy from raw image metadata
            img_5d = BioImage(df.raw_filepath.iloc[index], reader=bioio_nd2.Reader)
            ZvX = np.round(img_5d.physical_pixel_sizes.Z/img_5d.physical_pixel_sizes.X,0)
            anisotropy = (ZvX,1,1)
            # Load the denoised data
            img_5d = BioImage(df.denoised_filepath.iloc[index], reader=bioio_tifffile.Reader)
            for t in range(img_5d.dims.T):
                img = img_5d.get_image_data("CZYX", T=t)

                # Segment nuclei 
                masks,flows,styles = segment_nuclei(img[nucChannel,:,:,:],model) # Run the segmentation
                plot_qc_segmentation_xyz(img,masks,index, df, t, display_plot = False)                         # Create qc plot
                OmeTiffWriter.save(masks, output_path+'segmentation/'+df.id.iloc[index]+'_t'+'{:02d}'.format(t)+'.tif')

                del flows
                del styles
                gc.collect()
                
                # Calculate edt 
                masks_edt = calc_distance_mask(masks,anisotropy)
                OmeTiffWriter.save(masks_edt, output_path+'edt/'+df.id.iloc[index]+'_t'+'{:02d}'.format(t)+'.tif')

                del masks
                del masks_edt
                gc.collect()
                continue


## read images
## crop the nuclei slices
## calculate EDT transform
## for each nuclei loop over all distances (1:40) and take mean
## array of distance/intensity measurements are taken only for middle slice of mask (?)

## nucleus_id | nucleus volume | [1:20] mean intensities | group | ...


def run_dist_analysis(indices,df):
    '''Run the distance analysis on all images in the dataframe'''
    for index in tqdm.tqdm(indices):
        
        df_nuclei = pd.DataFrame()
        print(df.iloc[index].raw_filepath)

        img_5d = BioImage(df.raw_filepath.iloc[index], reader=bioio_nd2.Reader)
        # calculate anisotropy from raw image metadata
        ZvX = np.round(img_5d.physical_pixel_sizes.Z/img_5d.physical_pixel_sizes.X,0)

        for t in range(img_5d.dims.T):
            img = img_5d.get_image_data("ZYXC", T=t)

            masks = BioImage(output_path+'segmentation/'+df.id.iloc[index]+'_t'+'{:02d}'.format(t)+'.tif', reader=bioio_tifffile.Reader)
            masks = masks.get_image_data("ZYX", T=0, C=0)
            
            df_region_props = regionprops_table(masks,img, properties = ['label', 'area','centroid','MajorAxisLength','solidity','image','intensity_image'])
            df_region_props = pd.DataFrame(df_region_props)

            if len(df_region_props)>=10:
                plot_qc_nuclei_crop(df, index, df_region_props, img, t=t, display = False) 

            for i in range(len(df_region_props)):
                df_nuclei_temp = pd.DataFrame()

                intensity_image_spots = df_region_props['intensity_image'][i][:,:,:,spotChannel] #show spot channel
                intensity_image_nuclei = df_region_props['intensity_image'][i][:,:,:,nucChannel] #show nuclear ring channel

                image = df_region_props['image'][i]  # binary 3d mask

                # Extract the intensity per distance
                mx_spots = np.ma.masked_array(intensity_image_spots, mask = ~image) # 3d masked spot channel
                mx_nuclei = np.ma.masked_array(intensity_image_nuclei,mask = ~image) # 3d masked nuclear ring channel
                mx_mask = np.ma.masked_array(image,mask = ~image)  # 3d masked binary mask

                z_height = image.shape[0]

                slice_spots = mx_spots[int(z_height/2)]
                slice_nuclei = mx_nuclei[int(z_height/2)]
                slice_mask = mx_mask[int(z_height/2)]

                slice_mask_edt = edt.edt(slice_mask)
                slice_mask_edt = np.ma.masked_array(slice_mask_edt, mask = ~(slice_mask_edt>0)) 

                results = regionprops_table(slice_mask_edt.astype('int'),slice_nuclei,properties=['label','intensity_mean'])
                intensity_dist_nuclei = results['intensity_mean']

                results = regionprops_table(slice_mask_edt.astype('int'),slice_spots,properties=['label','intensity_mean'])
                intensity_dist_spots = results['intensity_mean']

                dist = results['label']

                df_nuclei_temp['label']  = [df_region_props.label.iloc[i]]
                df_nuclei_temp['bb_dimZ']  = [mx_spots.shape[0]]
                df_nuclei_temp['bb_dimY']  = [mx_spots.shape[1]]
                df_nuclei_temp['bb_dimX']  = [mx_spots.shape[2]]
                df_nuclei_temp['centroid_z'] = df_region_props['centroid-0'][i]
                df_nuclei_temp['centroid_y'] = df_region_props['centroid-1'][i]
                df_nuclei_temp['centroid_x'] = df_region_props['centroid-2'][i]
                df_nuclei_temp['major_axis_length'] = df_region_props['MajorAxisLength'][i]
                df_nuclei_temp['solidity'] = df_region_props['solidity'][i]
                df_nuclei_temp['mean'] = [np.ma.mean(mx_spots)]
                df_nuclei_temp['median'] = [np.ma.median(mx_spots)]
                df_nuclei_temp['std']=  [np.ma.std(mx_spots)]
                df_nuclei_temp['sum']= [np.ma.sum(mx_spots)]
                df_nuclei_temp['variance']= [np.ma.var(mx_spots)]
                df_nuclei_temp['max'] = [np.ma.max(mx_spots)]
                df_nuclei_temp['min'] = [np.ma.min(mx_spots)]
                df_nuclei_temp['volume'] = [np.sum(np.invert(mx_spots.mask))]
                df_nuclei_temp['id'] = [df.id.iloc[index]]
                df_nuclei_temp['timepoint'] = [t]
                df_nuclei_temp['intensity_dist_nuclei'] = [intensity_dist_nuclei]  # this is the emerin ring channel intensity on central slice
                df_nuclei_temp['intensity_dist_spots'] = [intensity_dist_spots] # this is the spot channel but not actual detected spots
                df_nuclei_temp['intensity_dist'] = [dist]  # this is the distance from the edge of the nucleus
                df_nuclei_temp['zproj_spots'] = [np.max(intensity_image_spots[:,:,:], axis = 0)]
                df_nuclei_temp['zproj_nuclei'] = [np.max(intensity_image_nuclei[:,:,:], axis = 0)]
                df_nuclei_temp['anisotropy'] = [ZvX]

                df_nuclei = pd.concat([df_nuclei,df_nuclei_temp])

        # save as pickle because has array stored in Dataframe
        df_nuclei.to_pickle(output_path+'dist/'+df.id.iloc[index]+'.pkl') # Back up the DF for this FOV

        # save with metadata as csv for simple viewing 
        df_nuclei_for_csv = pd.merge(df_nuclei,df,on='id',how='left')
        df_nuclei_for_csv.drop( columns = [ 'intensity_dist_nuclei','intensity_dist_spots','intensity_dist' ], axis=1, inplace=True)
        df_nuclei_for_csv.to_csv(output_path+'nuclei/'+df.id.iloc[index]+'.csv', index=False)


# Collect the data

def collect_nuclear_segmentation_data(indices, df):
    '''Collects nuclear data from Position-specific directories'''
    df_nuclei = pd.DataFrame()
    
    for index in tqdm.tqdm(indices):
        position_id = df.id.iloc[index]  # Should be "Position_X"
        
        # Path to CSV in Position-specific directory
        csv_path = os.path.join(
            output_path, 
            position_id,  # Position_X folder
            'nuclei',     # Subdirectory for nuclear measurements
            f"{position_id}.csv" 
        )
        
        if os.path.exists(csv_path):
            df_tmp = pd.read_csv(csv_path)
            df_nuclei = pd.concat([df_nuclei, df_tmp])
        else:
            print(f"Warning: Missing {csv_path}")

    # Save combined data
    output_file = os.path.join(output_path, f'nuclei_analysis.csv')
    df_nuclei.to_csv(output_file, index=False)

def collect_nuclear_distance_data(indices, df):
    '''Collects nuclear intensity and intensity vs distance data for all nuclei in the dataset'''
    df_dist = pd.DataFrame()
    for index in tqdm.tqdm(indices):
        df_tmp = pd.read_pickle(output_path+'dist/'+df.id.iloc[index]+'.pkl')
        df_dist = pd.concat([df_dist,df_tmp])
    df_dist.to_pickle(output_path+'dist_analysis_.pkl')