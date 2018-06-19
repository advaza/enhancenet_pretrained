#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import glob
import imageio
import easyargs
import cv2
import progressbar
from nnlib import *

def upscale_function(image):
    PER_CHANNEL_MEANS = np.array([0.47614917, 0.45001204, 0.40904046])
    # fns = sorted([fn for fn in os.listdir('input/')])
    # if not os.path.exists('output'):
    #     os.makedirs('output')
    # for fn in fns:
        # fne = ''.join(fn.split('.')[:-1])
        # if os.path.isfile('output/%s-EnhanceNet.png' % fne):
        #     print('skipping %s' % fn)
        #     continue
    # imgs = image
    # if imgs is None:
    #     continue
    imgs = np.expand_dims(image, axis=0)
    imgsize = np.shape(imgs)[1:]
    # print('processing %s' % fn)
    xs = tf.placeholder(tf.float32, [1, imgsize[0], imgsize[1], imgsize[2]])
    rblock = [resi, [[conv], [relu], [conv]]]
    ys_est = NN('generator',
                [xs,
                 [conv], [relu],
                 rblock, rblock, rblock, rblock, rblock,
                 rblock, rblock, rblock, rblock, rblock,
                 [upsample], [conv], [relu],
                 [upsample], [conv], [relu],
                 [conv], [relu],
                 [conv, 3]])
    ys_res = tf.image.resize_images(xs, [4*imgsize[0], 4*imgsize[1]],
                                    method=tf.image.ResizeMethod.BICUBIC)
    ys_est += ys_res + PER_CHANNEL_MEANS
    sess = tf.InteractiveSession()
    tf.train.Saver().restore(sess, os.getcwd()+'/weights')
    # output = sess.run([ys_est, ys_res+PER_CHANNEL_MEANS],
    #                   feed_dict={xs: imgs-PER_CHANNEL_MEANS})
    output = sess.run([ys_est, ys_res + PER_CHANNEL_MEANS],
                      feed_dict={xs: imgs - PER_CHANNEL_MEANS})
    # saveimg(output[0][0], 'output/%s-EnhanceNet.png' % fne)
    # saveimg(output[1][0], 'output/%s-Bicubic.png' % fne)
    sess.close()
    tf.reset_default_graph()
    return output[0][0]


def folders_in(directory, subfolder, recursive=True):
    # silly hack to handle file streams which respond only after query
    _ = glob.glob(os.path.join(directory, '**'), recursive=recursive)
    ret = [name for name in glob.glob(os.path.join(directory, '**'), recursive=recursive) if
            os.path.isdir(name)]
    return ret


def files_in(directory, extensions, recursive=False):
    return [name for name in glob.glob(os.path.join(directory, '**'), recursive=recursive) if
            os.path.splitext(name)[-1].lower() in extensions and '_EnhanceNet' not in name]


def process_image(image):

    image = image.astype(np.float32) / 255
    output_image = upscale_function(image)
    output_image = (output_image * 255).astype(np.uint8)
    return output_image


def process_out_file_path(file_name, output_dir):
    if output_dir is None:
        output_dir = os.path.abspath(os.path.dirname(file_name))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    basename = os.path.basename(file_name)
    extension = basename.split('.')[-1]
    out_name = basename[:-len(extension) - 1] + '_EnhanceNet' + '.' + extension

    return os.path.join(output_dir, out_name)


@easyargs
def main(in_folder=".", output_dir=None, in_subfolder=None):
    """
    Calculate histogram transfer from reference image to a given video
    :param in_folder: Input folder of folders with video files
    :return:
    """
    for folder in sorted(folders_in(in_folder, in_subfolder, recursive=True)):

        video_files = files_in(folder, extensions=['.mp4'])
        image_files = files_in(folder, extensions=['jpg', 'JPG', 'png', 'jpeg', 'JPEG'])

        if image_files:
            for image_file in image_files:

                out_file = process_out_file_path(image_file, output_dir)
                image = cv2.imread(image_file)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                out_image = process_image(image)
                out_image = cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(out_file, out_image)

        if video_files:
            for video_file in video_files:

                video_reader = imageio.get_reader(video_file)
                out_video = process_out_file_path(video_file, output_dir)
                writer = imageio.get_writer(out_video, fps=video_reader.get_meta_data()['fps'])
                print('Working on %s' % out_video)

                bar = progressbar.ProgressBar()
                for frame in bar(video_reader):

                    writer.append_data(process_image(frame))

                writer.close()


if __name__ == '__main__':
    main()

