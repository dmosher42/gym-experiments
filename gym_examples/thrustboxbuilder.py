# import xml.etree.ElementTree as ET
from pathlib import Path
import mujoco
import mujoco.viewer
import xml.etree.ElementTree as ET

import time

class Body:
	def __init__(self,pos:tuple[float,float,float],size:tuple[float,float,float]):
		inputtuple_1=("x","y","z")
		self.pos = dict(zip(inputtuple_1, pos))
		self.size = dict(zip(inputtuple_1, size))
		# self.color= dict(zip(("r","g","b","a"),rgba))

class Arms:
	"""

	"""
	def __init__(self):

		return

class Thrusters:

	def __init__(self):

		return

class Client:
	def __init__(self):
		return

class ThrustboxBuilder:
	"""
	## Description
	Script that builds Thrustbox environments for use with Gymnasium
	specs -> ThrustboxBuilder -> XML (MCJF) tempfile -> Gymnasium/mj_loadXML
	name: name of the outputted MuJoCo model

	Does not do action space or obsevation space dim matching
	TODO: Dim matching
	"""
	def __init__(self,name,preset="None"):
		self.currentdir = Path.cwd()
		if preset == "None":
			self.name = f"{name}"
		else:
			self.name = f"{preset}_{name}"
		self.box_size = (.2,.2,.2)
		self.arms = False
		self.armsoffset = .02

	def	boxsize(self,scale:float=(.2,.2,.2)):
		"""
		Sets the size of the satelite body
		:param scale: tuple. (X,Y,Z)
		:return: None
		"""
		self.box_size=scale

	def setthrusttype(self,thruster_type="simple"):
		"""
		Sets the type of thruster model to be used. NOTE: Will change the action space dims.
		:param simple: Sets the type of thruster action model to be used.
		Options are "simple" for full, simple cartesian (6 dim),
					"complex" for full, complex 3 thruster-on-each-corner setup (24 dim),
		:return: None
		"""
		self.thrustertype=thruster_type

	def setarms(self,length=.4):
		"""

		:return:
		"""
		self.armslength=length
		self.arms=True

	def makexml(self):
		# Starter XML to build the MuJoCo scene

		# TODO: there's a better way to do this.
		if self.thrustertype == "simple":
			thrustsnippet = f"""
			<site name="motorX" fromto="-{self.box_size[0]/2}  .0  .0  {self.box_size[0]/2}  .0  .0" class="thruster" rgba="1 0 0 1"/>
			<site name="motorY" fromto=" .0 -{self.box_size[1]/2}  .0  .0  {self.box_size[1]/2}  .0" class="thruster" rgba="0 1 0 1"/>
			<site name="motorZ" fromto=" .0  .0 -{self.box_size[2]/2}  .0  .0  {self.box_size[2]/2}" class="thruster" rgba="0 0 1 1"/>
			"""
			thrusteractuatorsnippet = """
			<motor name="ForceX" ctrllimited="true" ctrlrange="-1. 1.0" gear=".0  .0 -.2  0. 0. 0." site="motorX"/>
			<motor name="ForceY" ctrllimited="true" ctrlrange="-1. 1.0" gear=".0  .0 -.2  0. 0. 0." site="motorY"/>
			<motor name="ForceZ" ctrllimited="true" ctrlrange="-1. 1.0" gear=".0  .0 -.2  0. 0. 0." site="motorZ"/>

			<motor name="TorqueX" ctrllimited="true" ctrlrange="-1. 1.0" gear=".0  .0 .0  0. 0. -.01" site="motorX"/>
			<motor name="TorqueY" ctrllimited="true" ctrlrange="-1. 1.0" gear=".0  .0 .0  0. 0. -.01" site="motorY"/>
			<motor name="TorqueZ" ctrllimited="true" ctrlrange="-1. 1.0" gear=".0  .0 .0  0. 0. -.01" site="motorZ"/>
			"""
		else:
			thrustsnippet = f"""
			<site name="motor+Z+x+y" fromto=" {self.box_size[0]-.02}  {self.box_size[1]-.02}  {self.box_size[2]}  {self.box_size[0]-.02}  {self.box_size[1]-.02}  {self.box_size[2]+.02}" class="thruster" rgba="1 1 1 1"/>
			<site name="motor+Z-x+y" fromto="-{self.box_size[0]-.02}  {self.box_size[1]-.02}  {self.box_size[2]} -{self.box_size[0]-.02}  {self.box_size[1]-.02}  {self.box_size[2]+.02}" class="thruster" rgba="0 1 1 1"/>
			<site name="motor+Z-x-y" fromto="-{self.box_size[0]-.02} -{self.box_size[1]-.02}  {self.box_size[2]} -{self.box_size[0]-.02} -{self.box_size[1]-.02}  {self.box_size[2]+.02}" class="thruster" rgba="0 0 1 1"/>
			<site name="motor+Z+x-y" fromto=" {self.box_size[0]-.02} -{self.box_size[1]-.02}  {self.box_size[2]}  {self.box_size[0]-.02} -{self.box_size[1]-.02}  {self.box_size[2]+.02}" class="thruster" rgba="1 0 1 1"/>
			<!-- Thruster geom that are pointing in -z direction -->
			<site name="motor-Z+x+y" fromto=" {self.box_size[0]-.02}  {self.box_size[1]-.02} -{self.box_size[2]}  {self.box_size[0]-.02}  {self.box_size[1]-.02} -{self.box_size[2]+.02}" class="thruster" rgba="1 1 0 1"/>
			<site name="motor-Z-x+y" fromto="-{self.box_size[0]-.02}  {self.box_size[1]-.02} -{self.box_size[2]} -{self.box_size[0]-.02}  {self.box_size[1]-.02} -{self.box_size[2]+.02}" class="thruster" rgba="0 1 0 1"/>
			<site name="motor-Z-x-y" fromto="-{self.box_size[0]-.02} -{self.box_size[1]-.02} -{self.box_size[2]} -{self.box_size[0]-.02} -{self.box_size[1]-.02} -{self.box_size[2]+.02}" class="thruster" rgba="0 0 0 1"/>
			<site name="motor-Z+x-y" fromto=" {self.box_size[0]-.02} -{self.box_size[1]-.02} -{self.box_size[2]}  {self.box_size[0]-.02} -{self.box_size[1]-.02} -{self.box_size[2]+.02}" class="thruster" rgba="1 0 0 1"/>
			<!-- Thruster geom that are pointing in +y direction -->
			<site name="motor+Y+x+z" fromto=" {self.box_size[0]-.02}  {self.box_size[1]}  {self.box_size[2]-.02}  {self.box_size[0]-.02}  {self.box_size[1]+.02}  {self.box_size[2]-.02}" class="thruster" rgba="1 1 1 1"/>
			<site name="motor+Y+x-z" fromto=" {self.box_size[0]-.02}  {self.box_size[1]} -{self.box_size[2]-.02}  {self.box_size[0]-.02}  {self.box_size[1]+.02} -{self.box_size[2]-.02}" class="thruster" rgba="1 1 0 1"/>
			<site name="motor+Y-x-z" fromto="-{self.box_size[0]-.02}  {self.box_size[1]} -{self.box_size[2]-.02} -{self.box_size[0]-.02}  {self.box_size[1]+.02} -{self.box_size[2]-.02}" class="thruster" rgba="0 1 0 1"/>
			<site name="motor+Y-x+z" fromto="-{self.box_size[0]-.02}  {self.box_size[1]}  {self.box_size[2]-.02} -{self.box_size[0]-.02}  {self.box_size[1]+.02}  {self.box_size[2]-.02}" class="thruster" rgba="0 1 1 1"/>
			<!-- Thruster geom that are pointing in -y direction -->
			<site name="motor-Y+x+z" fromto=" {self.box_size[0]-.02} -{self.box_size[1]}  {self.box_size[2]-.02}  {self.box_size[0]-.02} -{self.box_size[1]+.02}  {self.box_size[2]-.02}" class="thruster" rgba="1 0 1 1"/>
			<site name="motor-Y+x-z" fromto=" {self.box_size[0]-.02} -{self.box_size[1]} -{self.box_size[2]-.02}  {self.box_size[0]-.02} -{self.box_size[1]+.02} -{self.box_size[2]-.02}" class="thruster" rgba="1 0 0 1"/>
			<site name="motor-Y-x-z" fromto="-{self.box_size[0]-.02} -{self.box_size[1]} -{self.box_size[2]-.02} -{self.box_size[0]-.02} -{self.box_size[1]+.02} -{self.box_size[2]-.02}" class="thruster" rgba="0 0 0 1"/>
			<site name="motor-Y-x+z" fromto="-{self.box_size[0]-.02} -{self.box_size[1]}  {self.box_size[2]-.02} -{self.box_size[0]-.02} -{self.box_size[1]+.02}  {self.box_size[2]-.02}" class="thruster" rgba="0 0 1 1"/>
			<!-- Thruster geom that are pointing in +x direction -->
			<site name="motor+X+y+z" fromto=" {self.box_size[0]}  {self.box_size[1]-.02}  {self.box_size[2]-.02}  {self.box_size[0]+.02}   {self.box_size[1]-.02}  {self.box_size[2]-.02}" class="thruster" rgba="1 1 1 1"/>
			<site name="motor+X-y+z" fromto=" {self.box_size[0]} -{self.box_size[1]-.02}  {self.box_size[2]-.02}  {self.box_size[0]+.02}  -{self.box_size[1]-.02}  {self.box_size[2]-.02}" class="thruster" rgba="1 0 1 1"/>
			<site name="motor+X-y-z" fromto=" {self.box_size[0]} -{self.box_size[1]-.02} -{self.box_size[2]-.02}  {self.box_size[0]+.02}  -{self.box_size[1]-.02} -{self.box_size[2]-.02}" class="thruster" rgba="1 0 0 1"/>
			<site name="motor+X+y-z" fromto=" {self.box_size[0]}  {self.box_size[1]-.02} -{self.box_size[2]-.02}  {self.box_size[0]+.02}   {self.box_size[1]-.02} -{self.box_size[2]-.02}" class="thruster" rgba="1 1 0 1"/>
			<!-- Thruster geom that are pointing in -x direction -->
			<site name="motor-X+y+z" fromto="-{self.box_size[0]}  {self.box_size[1]-.02}  {self.box_size[2]-.02} -{self.box_size[0]+.02}   {self.box_size[1]-.02}  {self.box_size[2]-.02}" class="thruster" rgba="0 1 1 1"/>
			<site name="motor-X-y+z" fromto="-{self.box_size[0]} -{self.box_size[1]-.02}  {self.box_size[2]-.02} -{self.box_size[0]+.02}  -{self.box_size[1]-.02}  {self.box_size[2]-.02}" class="thruster" rgba="0 0 1 1"/>
			<site name="motor-X-y-z" fromto="-{self.box_size[0]} -{self.box_size[1]-.02} -{self.box_size[2]-.02} -{self.box_size[0]+.02}  -{self.box_size[1]-.02} -{self.box_size[2]-.02}" class="thruster" rgba="0 0 0 1"/>
			<site name="motor-X+y-z" fromto="-{self.box_size[0]}  {self.box_size[1]-.02} -{self.box_size[2]-.02} -{self.box_size[0]+.02}   {self.box_size[1]-.02} -{self.box_size[2]-.02}" class="thruster" rgba="0 1 0 1"/>
		"""
			thrusteractuatorsnippet = """
			<motor name="1X(white)" ctrllimited="true" ctrlrange="0.0 1.0" gear=".0  0. .2  0. 0. 0." site="motor+X+y+z"/>
			<motor name="1Y(white)" ctrllimited="true" ctrlrange="0.0 1.0" gear=".0  0. .2  0. 0. 0." site="motor+Y+x+z"/>
			<motor name="1Z(white)" ctrllimited="true" ctrlrange="0.0 1.0" gear=".0  0. .2  0. 0. 0." site="motor+Z+x+y"/>
			<motor name="2X(cyan)"  ctrllimited="true" ctrlrange="0.0 1.0" gear=".0  0. .2  0. 0. 0." site="motor-X+y+z"/>
			<motor name="2Y(cyan)"  ctrllimited="true" ctrlrange="0.0 1.0" gear=".0  0. .2  0. 0. 0." site="motor+Y-x+z"/>
			<motor name="2Z(cyan)"  ctrllimited="true" ctrlrange="0.0 1.0" gear=".0  0. .2  0. 0. 0." site="motor+Z-x+y"/>
			<motor name="3X(blue)"  ctrllimited="true" ctrlrange="0.0 1.0" gear=".0  0. .2  0. 0. 0." site="motor-X-y+z"/>
			<motor name="3Y(blue)"  ctrllimited="true" ctrlrange="0.0 1.0" gear=".0  0. .2  0. 0. 0." site="motor-Y-x+z"/>
			<motor name="3Z(blue)"  ctrllimited="true" ctrlrange="0.0 1.0" gear=".0  0. .2  0. 0. 0." site="motor+Z-x-y"/>
			<motor name="4X(purple)" ctrllimited="true" ctrlrange="0.0 1.0" gear=".0  0. .2  0. 0. 0." site="motor+X-y+z"/>
			<motor name="4Y(purple)" ctrllimited="true" ctrlrange="0.0 1.0" gear=".0  0. .2  0. 0. 0." site="motor-Y+x+z"/>
			<motor name="4Z(purple)" ctrllimited="true" ctrlrange="0.0 1.0" gear=".0  0. .2  0. 0. 0." site="motor+Z+x-y"/>
			<motor name="5X(yellow)" ctrllimited="true" ctrlrange="0.0 1.0" gear=".0  0. .2  0. 0. 0." site="motor+X+y-z"/>
			<motor name="5Y(yellow)" ctrllimited="true" ctrlrange="0.0 1.0" gear=".0  0. .2  0. 0. 0." site="motor+Y+x-z"/>
			<motor name="5Z(yellow)" ctrllimited="true" ctrlrange="0.0 1.0" gear=".0  0. .2  0. 0. 0." site="motor-Z+x+y"/>
			<motor name="6X(green)"  ctrllimited="true" ctrlrange="0.0 1.0" gear=".0  0. .2  0. 0. 0." site="motor-X+y-z"/>
			<motor name="6Y(green)"  ctrllimited="true" ctrlrange="0.0 1.0" gear=".0  0. .2  0. 0. 0." site="motor+Y-x-z"/>
			<motor name="6Z(green)"  ctrllimited="true" ctrlrange="0.0 1.0" gear=".0  0. .2  0. 0. 0." site="motor-Z-x+y"/>
			<motor name="7X(black)"  ctrllimited="true" ctrlrange="0.0 1.0" gear=".0  0. .2  0. 0. 0." site="motor-X-y-z"/>
			<motor name="7Y(black)"  ctrllimited="true" ctrlrange="0.0 1.0" gear=".0  0. .2  0. 0. 0." site="motor-Y-x-z"/>
			<motor name="7Z(black)"  ctrllimited="true" ctrlrange="0.0 1.0" gear=".0  0. .2  0. 0. 0." site="motor-Z-x-y"/>
			<motor name="8X(red)"    ctrllimited="true" ctrlrange="0.0 1.0" gear=".0  0. .2  0. 0. 0." site="motor+X-y-z"/>
			<motor name="8Y(red)"    ctrllimited="true" ctrlrange="0.0 1.0" gear=".0  0. .2  0. 0. 0." site="motor-Y+x-z"/>
			<motor name="8Z(red)"    ctrllimited="true" ctrlrange="0.0 1.0" gear=".0  0. .2  0. 0. 0." site="motor-Z+x-y"/>
			"""
		if self.arms:
			armsactuatorsnippet = f"""
		<motor ctrllimited="true" ctrlrange="-1 0" joint="shoulder_1" gear="150"/>
		<motor ctrllimited="true" ctrlrange="-1 0" joint="fore_1" gear="150"/>
		<motor ctrllimited="true" ctrlrange="-1 0" joint="shoulder_2" gear="150"/>
		<motor ctrllimited="true" ctrlrange="-1 0" joint="fore_2" gear="150"/>
"""
			armssnippet = f"""
			<body name="aux_1" pos="{self.box_size[0]} 0 -{self.box_size[2]-self.armsoffset}">
				<joint axis="0 1 0" name="shoulder_1" type="hinge" class="shoulder"/>
				<geom fromto="0.0 0.0 0.0 {self.armslength} 0. 0.0" name="left_arm_geom" class="armgeom"/>
				<body pos="{self.armslength} 0. 0">
					<joint axis="0 1 0" name="fore_1" type="hinge" class="elbow"/>
					<geom fromto="0.0 0.0 0.0 {self.armslength} 0. 0.0" name="left_fore_geom" class="armgeom"/>
				</body>
			</body>
			<body name="aux_2" pos="-{self.box_size[0]} 0 -{self.box_size[2]-self.armsoffset}">
				<joint axis="0 -1 0" name="shoulder_2" type="hinge" class="shoulder"/>
				<geom fromto="0.0 0.0 0.0 -{self.armslength} -0. 0.0" name="right_arm_geom" class="armgeom"/>
				<body pos="-{self.armslength} -0. 0">
					<joint axis="0 -1 0" name="fore_2" type="hinge" class="elbow"/>
					<geom fromto="0.0 0.0 0.0 -{self.armslength} -0. 0.0" name="right_fore_geom" class="armgeom"/>
				</body>
				<!-- </body> -->
			</body>
"""
		else:
			armssnippet= ""
			armsactuatorsnippet = ""

		basexml = f"""
		<mujoco model="{self.name}">
			<compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
			<option timestep="0.01" gravity="0 0 0" density="1" viscosity="0" integrator="RK4"/>
			<visual>
				<map force="0.1" zfar="30"/>
				<rgba haze="0.15 0.25 0.35 1"/>
				<global offwidth="2560" offheight="1440" elevation="-20" azimuth="120"/>
			</visual>
			<default>
				<joint armature="1" damping="1" limited="true"/>
				<geom conaffinity="1" condim="1" density="5.0" friction="1 0.5 0.5" margin="0.01" rgba="0.8 0.6 0.4 1"/>
				<default class="shoulder">
					<joint pos="0.0 0.0 0.0" stiffness="100" range="30 70" springref="70"/>
				</default>
				<default class="armgeom">
					<geom size="0.08" type="capsule" rgba="1 .6 0 1"/>
				</default>
				<default class="elbow">
					<joint pos="0.0 0.0 0.0" stiffness="100" range="30 70" springref="70"/>
				</default>
				<default class="thruster">
					<site type="cylinder" size="0.02 0.05" rgba="0.3 0.8 0.3 1" quat="1.0 0.0 0.0 0."/>
				</default>
			</default>
			<asset>
				<texture type="skybox" builtin="gradient" rgb1=".3 .5 .7" rgb2="0 0 0" width="32" height="512"/>
			</asset>
			<worldbody>
				<!-- <geom name="floor" pos = "0 0 0" size="2 2 .125" type="plane" conaffinity="1" condim="3"/> -->
				<!-- Satellite -->
				<body name="box" pos="0 0 1">
					<geom name="core" pos="0 0 0" quat="1 0 0 0" size="{self.box_size[0]} {self.box_size[1]} {self.box_size[2]}" type="box" rgba="0.3 0.3 0.8 .5" mass=".1"/>
					<joint name="root" type="free" damping="0" armature="0" pos="0 0 0"/>
					<site name="IMUmount" pos="0.0 0.0 0.0" type="sphere" size="0.02" quat="1.0 0.0 0.0 0." rgba="1 1 1 1"/>

					{thrustsnippet}
					{armssnippet}
					<!-- Visualization of the coordinate frame-->
					<site name="qcX" type="box" pos="0.2 0.0 0.0" size="0.2 0.005 0.005" quat=" 1.000  0.0  0.0    0." rgba="1 0 0 1"/>
					<site name="qcY" type="box" pos="0.0 0.2 0.0" size="0.2 0.005 0.005" quat=" 0.707  0.0  0.0    0.707" rgba="0 1 0 1"/>
					<site name="qcZ" type="box" pos="0.0 0.0 0.2" size="0.2 0.005 0.005" quat="-0.707  0.0  0.707  0." rgba="0 0 1 1"/>
					<!--
					<site name="motor0" type="cylinder" size="0.1 0.05"  fromto = ".0 0 .2 .0 .0 .25"quat = "1.0 0.0 0.0 0." rgba="0.3 0.8 0.3 1"/>
					<site name="motor1" type="cylinder" size="0.1 0.05"  fromto = ".0 .2 .0 .0 .25 .0"  quat = "1.0 0.0 0.0 0." rgba="0.3 0.8 0.3 1"/>
					<site name="motor2" type="cylinder" size="0.1 0.05"  fromto = ".2 .0 .0 .25 0 .0"  quat = "1.0 0.0 0.0 0." rgba="0.3 0.8 0.3 1"/>
					<site name="motor3" type="cylinder" size="0.1 0.05"  fromto = "0 -.2 .0 .0 -.25 .0"  quat = "1.0 0.0 0.0 0." rgba="0.3 0.8 0.3 1"/>
					<site name="motor4" type="cylinder" size="0.1 0.05"  fromto = "-.2 .0 .0 -.25 0 .0"  quat = "1.0 0.0 0.0 0." rgba="0.3 0.8 0.3 1"/>
					-->
					<!-- 			<body name="left_arm" pos="0 0 0">
						<geom fromto="0.0 0.0 -0.18 0.2 0 -0.18" name="aux_1_geom" class ="armgeom"/> -->
				</body>
				<!-- Target -->
				<body name="target" pos=".1 -.1 .01">
					<camera name = "camera1" mode = "targetbodycom" target = "world" pos="2.370 -2.859 2.423" xyaxes="0.770 0.638 0.000 -0.228 0.275 0.934"/>
					<joint armature="0" axis="1 0 0" damping="0" limited="true" name="target_x" pos="0 0 0" range="-3 3" ref="0" stiffness="0" type="slide"/>
					<joint armature="0" axis="0 1 0" damping="0" limited="true" name="target_y" pos="0 0 0" range="-3 3" ref="0" stiffness="0" type="slide"/>
					<joint armature="0" axis="0 0 1" damping="0" limited="true" name="target_z" pos="0 0 0" range="-3 3" ref="0" stiffness="0" type="slide"/>
					<geom conaffinity="0" contype="0" name="target" pos="0 0 0" rgba="0.9 0.2 0.2 1" size=".05" type="sphere" mass=".1"/>

				</body>
			</worldbody>
			<actuator>
			{thrusteractuatorsnippet}
			{armsactuatorsnippet}
			</actuator>

			<sensor>
				<accelerometer name="accel" site = "IMUmount"/>
				<gyro name="gyro" site = "IMUmount"/>
			</sensor>
		</mujoco>

		 """

		print(basexml)
		root = ET.fromstring(basexml)
		# test = ET.Element("test")
		# root.find("worldbody").append(test)
		xmlstring = ET.tostring(root,method="xml")
		return xmlstring


if __name__ == '__main__':
	test = ThrustboxBuilder("simplethrustbox_target_Z_v0")
	test.setthrusttype("simple")
	test.boxsize((.2,.2,.7))
	# test.setarms()
	test.setarms(.4)

	xmlstring = test.makexml()
	print(xmlstring)

	m = mujoco.MjModel.from_xml_string(xmlstring)
	d = mujoco.MjData(m)


	with mujoco.viewer.launch_passive(m, d) as viewer:
		# Close the viewer automatically after 30 wall-seconds.
		start = time.time()
		while viewer.is_running() and time.time() - start < 30:
			step_start = time.time()

			# mj_step can be replaced with code that also evaluates
			# a policy and applies a control signal before stepping the physics.
			mujoco.mj_step(m, d)

			# Example modification of a viewer option: toggle contact points every two seconds.
			with viewer.lock():
				viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

			# Pick up changes to the physics state, apply perturbations, update options from GUI.
			viewer.sync()

			# Rudimentary time keeping, will drift relative to wall clock.
			time_until_next_step = m.opt.timestep - (time.time() - step_start)
			if time_until_next_step > 0:
				time.sleep(time_until_next_step)