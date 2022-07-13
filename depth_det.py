import time
import pandas as pd
import os
import PIL
import cv2
from tqdm import tqdm

import sys
sys.path.append('../')

from pipeline_input import *
from constants import *

class depth_interp_airsim(pipeline_dataset_interpreter):
	NUM_CAMS = 0
	mode_name = {
        0: 'Scene', 
        1: 'DepthPlanar', 
        2: 'DepthPerspective',
        3: 'DepthVis', 
        4: 'DisparityNormalized',
        5: 'Segmentation',
        6: 'SurfaceNormals',
        7: 'Infrared'
    }
	cam_name = {
        '0': 'FrontL'
    }

	def load(self) -> None:
		super().load()
		airsim_txt=os.path.join(self.input_dir, 'airsim_rec.txt')
		images_folder=os.path.join(self.input_dir, 'images')
		
		assert os.path.exists(airsim_txt)
		assert os.path.exists(images_folder)

		df = pd.read_csv(airsim_txt, sep='\t')
		df.set_index('TimeStamp')

		self.NUM_CAMS = len(df["ImageFile"].iloc[0].split(";"))

		self.cam_name = {
			'0': 'FrontL',
			str(self.NUM_CAMS): 'FrontR'
		}
		for i in range(1, self.NUM_CAMS):
			self.cam_name[str(i)] = 'C' + str(i)

		x_vals = dict()
		for col in df.columns:
			if col!="ImageFile":
				x_vals[col] = []
		x_vals["ImageFile"] = []

		for index, row in df.iterrows():
			files = row["ImageFile"].split(";")
			input_images = []
			gt_images = []
			for f in files:
				cam_id, img_format = f.split("_")[2:4]
				# if img_format in (1,2,3,4,5,6):
				# 	gt_images.append(os.path.join(images_folder, f))
				# 	assert os.path.exists(gt_images[-1])
				if img_format=='0':
					input_images.append(os.path.join(images_folder, f))
					assert os.path.exists(input_images[-1])
			for col in df.columns:
				if col!="ImageFile":
					x_vals[col] += [row[col]]
			x_vals["ImageFile"] += [";".join(input_images)]
		x_vals = pd.DataFrame(x_vals)
		self.dataset = {
			'train': {
				'x': x_vals,
				'y': df
			},
			'test': {
				'x': x_vals,
				'y': df
			}
		}


class videos_vis(pipeline_data_visualizer):

	def visualize(self, x, y, results, preds, directory) -> None:
		writer = None

		for frame in tqdm(preds):
			all_frames = []
			for index, row in frame.iterrows():
				f = row['input']
				input_img = cv2.imread(row['input_full'])
				depth = row['depth']
				cam_id, img_format = f.split("_")[2:4]
				full_frame = cv2.vconcat([input_img, depth])
				all_frames.append(full_frame)
				#cv2.imshow('depth_'+str(cam_id), full_frame)
			#cv2.imshow('depth', cv2.hconcat(all_frames))
			final_frame = cv2.hconcat([all_frames[0], all_frames[2], all_frames[3]])
			#cv2.imshow('depth', final_frame)
			if writer is None:
				size = final_frame.shape[:2]
				print("size:", size)
				output_path = os.path.join(directory, 'output.mp4')
				#writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (size[1],size[0]))
				writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'avc1'), 10, (size[1],size[0]))
			writer.write(final_frame)
			#time.sleep(0.3)
			#cv2.waitKey(1)
		writer.release()


class depth_evaluator:

	def evaluate(self, x: pd.DataFrame, y, plot=False):
		predict_results = []
		
		for inxex, row in tqdm(x.iterrows(), total=x.shape[0]):
			pred = self.predict(row)
			predict_results.append(pred)
		
		results = 0
		# TODO: implement evaluation
		return results, predict_results


	def evaluate_old(self, x: pd.DataFrame, y, plot=False):
		predict_results = {
			'input':[],'depth':[],'input_full':[]
		}
		for inxex, row in x.iterrows():
			pred = self.predict(row)
			for index2, row2 in pred.iterrows():
				predict_results['input'] += [row2['input']]
				predict_results['input_full'] += [row2['input_full']]
				predict_results['depth'] += [row2['depth']]
		
		predict_results = pd.DataFrame(predict_results)
		
		results = 0
		# TODO: implement evaluation
		return results, predict_results

class depth_pipeline_model(depth_evaluator, pipeline_model):

	def load(self):
		import monodepth2
		self.model = monodepth2.monodepth2()
		
	def train(self, x, y):
		predict_results = []
		for inxex, row in tqdm(x.iterrows(), total=x.shape[0]):
			pred = self.predict(row)
			predict_results.append(pred)
		
		results = 0
		# TODO: implement training
		return results, predict_results
		
	def predict(self, x) -> np.array:
		# Runs prediction on list of values x of length n
		# Returns a list of values of length n
		predict_results = {
			'input':[],'depth':[],'input_full':[]
		}
		# TODO: Implement prediction
		image_files = x["ImageFile"].split(";")
		for image_path in image_files:

			f = image_path.split("/")[-1]
			cam_id, img_format = f.split("_")[2:4]
			img_format = int(img_format)
			if f.endswith('.ppm'):
				#img = PIL.Image.open(image_path)
				#img = np.array(img.getdata()).reshape(img.size[1], img.size[0], 3)
				img = cv2.imread(image_path)
			#elif f.endswith('.pfm'):
			#	img, scale = airsim.read_pfm(f)
			else:
				print("Unknown format")

			depth = self.model.eval(img)
			predict_results['input'] += [f]
			predict_results['input_full'] += [image_path]
			predict_results['depth'] += [depth]
			
		# 	cv2.imshow('depth_'+str(cam_id),depth)
		# cv2.waitKey(1)
		predict_results = pd.DataFrame(predict_results)
		return predict_results

class depth_pipeline_monodepth2_mono_640x192(depth_pipeline_model, depth_evaluator, pipeline_model):

	def load(self):
		import monodepth2
		self.model = monodepth2.monodepth2(model_name='mono_640x192')

class depth_pipeline_monodepth2_stereo_640x192(depth_pipeline_model, depth_evaluator, pipeline_model):

	def load(self):
		import monodepth2
		self.model = monodepth2.monodepth2(model_name='stereo_640x192')

class depth_pipeline_monodepth2_mono_stereo_640x192(depth_pipeline_model, depth_evaluator, pipeline_model):

	def load(self):
		import monodepth2
		self.model = monodepth2.monodepth2(model_name='mono+stereo_640x192')


class depth_pipeline_monodepth2_mono_no_pt_640x192(depth_pipeline_model, depth_evaluator, pipeline_model):

	def load(self):
		import monodepth2
		self.model = monodepth2.monodepth2(model_name='mono_no_pt_640x192')


class depth_pipeline_monodepth2_stereo_no_pt_640x192(depth_pipeline_model, depth_evaluator, pipeline_model):

	def load(self):
		import monodepth2
		self.model = monodepth2.monodepth2(model_name='stereo_no_pt_640x192')


class depth_pipeline_monodepth2_mono_stereo_no_pt_640x192(depth_pipeline_model, depth_evaluator, pipeline_model):

	def load(self):
		import monodepth2
		self.model = monodepth2.monodepth2(model_name='mono+stereo_no_pt_640x192')


class depth_pipeline_monodepth2_mono_1024x320(depth_pipeline_model, depth_evaluator, pipeline_model):

	def load(self):
		import monodepth2
		self.model = monodepth2.monodepth2(model_name='mono_1024x320')


class depth_pipeline_monodepth2_stereo_1024x320(depth_pipeline_model, depth_evaluator, pipeline_model):

	def load(self):
		import monodepth2
		self.model = monodepth2.monodepth2(model_name='stereo_1024x320')


class depth_pipeline_monodepth2_mono_stereo_1024x320(depth_pipeline_model, depth_evaluator, pipeline_model):

	def load(self):
		import monodepth2
		self.model = monodepth2.monodepth2(model_name='mono+stereo_1024x320')


class depth_pipeline_monodepth2_mono_640x192(depth_pipeline_model, depth_evaluator, pipeline_model):

	def load(self):
		import monodepth2
		self.model = monodepth2.monodepth2(model_name='mono_640x192')



class depth_pipeline_manydepth(depth_pipeline_model, depth_evaluator, pipeline_model):

	def load(self):
		import manydepth
		self.model = manydepth.manydepth()


depth_input = pipeline_input("depth_det", {'depth_interp_airsim': depth_interp_airsim}, 
	{
		'depth_pipeline_monodepth2_mono_640x192': depth_pipeline_monodepth2_mono_640x192,
		'depth_pipeline_monodepth2_stereo_640x192': depth_pipeline_monodepth2_stereo_640x192,
		'depth_pipeline_monodepth2_mono_stereo_640x192': depth_pipeline_monodepth2_mono_stereo_640x192,
		'depth_pipeline_monodepth2_mono_no_pt_640x192': depth_pipeline_monodepth2_mono_no_pt_640x192,
		'depth_pipeline_monodepth2_stereo_no_pt_640x192': depth_pipeline_monodepth2_stereo_no_pt_640x192,
		'depth_pipeline_monodepth2_mono_stereo_no_pt_640x192': depth_pipeline_monodepth2_mono_stereo_no_pt_640x192,
		'depth_pipeline_monodepth2_mono_1024x320': depth_pipeline_monodepth2_mono_1024x320,
		'depth_pipeline_monodepth2_stereo_1024x320': depth_pipeline_monodepth2_stereo_1024x320,
		'depth_pipeline_monodepth2_mono_stereo_1024x320': depth_pipeline_monodepth2_mono_stereo_1024x320,
		#'depth_pipeline_manydepth': depth_pipeline_manydepth
	}, dict(), {
		'videos_vis': videos_vis
	})


exported_pipeline = depth_input
