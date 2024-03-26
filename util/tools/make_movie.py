import pyvista as pv
import os
import moviepy.video.io.ImageSequenceClip
import pdb

image_files_pressure = []
image_files_velocity = []
mesh = pv.read("data/movie_data/sten_flow_3.vtu")
for i in range(10):
    time_name = format(int(100*i), '05d')

    plotter = pv.Plotter(off_screen = True)
    slice = mesh.slice(normal=[1, 0, 1])
    plotter.add_mesh(mesh = slice, clim=[-30*1333, 1333*30], below_color='purple', above_color='yellow', scalars = f"pressure_{time_name}")
    image_path_pressure = f"data/movie_data/shot_pressure_{time_name}.png"
    plotter.screenshot(image_path_pressure)
    image_files_pressure.append(image_path_pressure)

    plotter = pv.Plotter(off_screen = True)
    slice = mesh.slice(normal=[1, 0, 1])
    plotter.add_mesh(mesh = slice, clim=[0, 600], below_color='purple', above_color='yellow', scalars = f"velocity_{time_name}")
    image_path_velocity = f"data/movie_data/shot_velocity_{time_name}.png"
    plotter.screenshot(image_path_velocity)
    image_files_velocity.append(image_path_velocity)

fps=2

clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files_velocity, fps=fps)
clip.write_videofile('data/movie_data/stenosis_velocity.mp4', codec="libx264")

clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files_pressure, fps=fps)
clip.write_videofile('data/movie_data/stenosis_pressure.mp4', codec="libx264")
# for i in range(10):


#     plotter = pv.Plotter(off_screen = True)
#     slice = mesh.slice(normal=[1, 0, 0])
#     plotter.add_mesh(mesh = slice, clim=[0, 6*10**3], below_color='purple', above_color='yellow', scalars = f"pressure_00{field_num}")
#     image_path = f"/home/nrubio/Desktop/SV_scripts/movie_images/shot_pressure_00{field_num}.png"
#     plotter.screenshot(image_path)

#     image_files.append(image_path)

# fps=5

# clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
# clip.write_videofile('/home/nrubio/Desktop/SV_scripts/geom0_transient_pressure.mp4')
