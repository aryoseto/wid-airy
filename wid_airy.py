# Wave-in-deck using Mercier Method 
# Airy wave theory
# This is a new branch

import sys
import numpy
import math
import matplotlib.pyplot as plt
from scipy.integrate import simps

# Airy wave, define all wave parameters
T = 13.1	# Wave period - seconds
H = 31.8	# Wave height - meters
D = 90.48	# Still water depth - meters

# Current Data
U_current = 0.93 # Maximum current 


# Deck location parameter
B = 44.75	# Topsides width, normal to wave - meters
Ld = 94.00	# Topsides length - meters
Hd = 14.77	# Bottom of topsides elevation (from d) - meters


# Constant
G = 9.81	# Gravity - m/s2
RHO = 1025	# Seawater density - kg/m3

###	--- Function Definition --- ###

# Calculate wave length
def wave_length (depth, period) :
	
	# Initial guess of wave length
	temp_L = G*period**2/(2*math.pi)
	
	# Calculate wave length based initial assumption
	L1 = temp_L * math.tanh(2 * math.pi * depth/temp_L)

	#count = 0

	# iterate until both giving the same result
	while abs(temp_L - L1) > 0.001 :
		
		#print( count, temp_L, L1)

		temp_L = (temp_L + L1)/2
		L1 = G*period**2/(2*math.pi) * math.tanh(2 * math.pi * depth/temp_L)
		
		#count += 1

	return round(L1,2)



# Calculate wave surface, relative to still water level (d)
def wave_surface (depth, period, height, length, time, loc):
	
	# surface elevation (from still water level)
	surface_el = height/2 * math.cos(2*math.pi*(loc/length - time/period)) 

	# global surface elevation relative to mudline
	#global_surface_el = surface_el + depth

	return round(surface_el,3)


# Horizontal particle velocity
def wave_hor_vel (depth, period, height, length, time, loc, surface_el) :

	velo_horizontal = (math.pi*height/period) * math.cosh((2*math.pi/length)*(surface_el+depth))/math.sinh(2*math.pi*depth/length) * math.cos(2*math.pi*(loc/length - time/period)) 

	return round(velo_horizontal,3)


# Vertical particle velocity
def wave_ver_vel (depth, period, height, length, time, loc, surface_el) :

	velo_vertical = (math.pi*height/period) * math.sinh((2*math.pi/length)*(surface_el+depth))/math.sinh(2*math.pi*depth/length) * math.sin(2*math.pi*(loc/length - time/period))

	return round(velo_vertical,3)


# To find t(s) when crest reach certain level at a location  
def find_t(depth, period, height, length, loc, surface_target, status) :
	
	# status positive means front of crest, negative means back of the crest
	if status > 0 :
		a_time = period*(loc/length - math.acos(2*surface_target/height)/(2*math.pi))
		return round(a_time,3)

	elif status < 0 :
		a_time = period*(loc/length + math.acos(2*surface_target/height)/(2*math.pi))
		return round(a_time,3)		


# Calculate WID - F front
def WID_front(vel_hori, vel_current, surface, topsides_width, topsides_BOS) :
	
	if surface > topsides_BOS :
		F_front = RHO * topsides_width * (vel_hori + vel_current)**2 * (surface - topsides_BOS)
		return round(F_front,3)

	else :
		return 0.00

# Defining the sample range, returns a list
def sample_range (starting, ending, size) :
	
	sample = []
	
	for step in numpy.arange(starting, ending, size) :
		sample.append(round(step,3))

	# Adding last step
	sample.append(ending)

	return sample





## -- Running Program --- ###

L = wave_length(D, T)
print("wave length :", L)

u = wave_hor_vel(D, T, H, L, 0, 0, H/2)
print("horizontal velocity :", u)

v = wave_ver_vel(D, T, H, L, 0, 0, H/2)
print("vertical velocity :", v)


# (depth, period, height, length, time, loc, surface_el)



# -- Testing output [START]
# # Horizonal Particle velocity for different location
# list_location = numpy.arange(-L/2, L/2, L/100)

# # Calculate wave surface with given location
# z_all = [] 
# for a_loc in list_location :
# 	z_all.append(wave_surface(D, T, H, L, 0, a_loc))

# # Only take local surface elevation
# z_local = []
# for item in z_all :
# 	z_local.append(item[0])

# # Calculate horizontal particle velocity at every location
# list_u = []
# for index in range(0, len(list_location)) :
# 	list_u.append(wave_hor_vel(D, T, H, L, 0, list_location[index], z_local[index]))

# # Calculate vertical particle velocity at every location 
# list_v =[]
# for index in range(0, len(list_location)) :
# 	list_v.append(wave_ver_vel(D, T, H, L, 0, list_location[index], z_local[index]))

# # Calculate t1, crest hit the front of the deck
# print(find_t(D, T, H, L, -Ld/2, Hd))



# print(list_u)

# plt.figure(1)
# plt.subplot(211)
# plt.plot(list_location, list_u ,list_location, list_v)

# plt.subplot(212)
# plt.plot(list_location, z_local)

# plt.show()

# -- Testing output [END]

# Check if inundation occurs
if H/2 > Hd :
	print ("Wave inundation : " + str(round(H/2 - Hd, 2)) + "m")

else:
	print ("No wave inundation")
	sys.exit()

# Calculate important t
# t = 0	: Max crest at x = 0 (centre)

# t1 	: When front of wave hits the deck
t1 = find_t(D, T, H, L, -Ld/2, Hd, 1)

# t2	: Wave crest at front of the deck (-Ld/2) 
t2 = find_t(D, T, H, L, -Ld/2, H/2, 1)

# t3	: Back of wave at front of the deck (-Ld/2)
t3 = find_t(D, T, H, L, -Ld/2, Hd, -1)

# t4	: Front of wave reaches back of the deck (Ld/2)
t4 = find_t(D, T, H, L, Ld/2, Hd, 1)

# t5	: Crest reaches back of the deck (Ld/2)
t5 = find_t(D, T, H, L, Ld/2, H/2, 1)

# t6	: Back of wave exiting at the back of the deck (Ld/2)
t6 = find_t(D, T, H, L, Ld/2, Hd, -1)

print(t1, t2, t3, t4, t5, t6)


# Define time range for calculating the WID forces
time_range = numpy.arange(t1, t6, (t6-t1)/100)


# Calculate WID - Ffront

# Calculate surface & horizontal particle velocity at front of deck (-Ld/2)
data_front = []
for time in time_range :
	
	surface_temp = wave_surface (D, T, H, L, time, -Ld/2)
	
	data_front.append([round(time, 3), surface_temp, wave_hor_vel(D, T, H, L, time, -Ld/2, surface_temp)])


#print(*data_front, sep="\n")

# Calculate Front force
force_front_list = []
for a_set in data_front :

	force_front_list.append(WID_front(a_set[2], U_current, a_set[1], B, Hd))

# print(*force_front_list, sep="\n")




# Calculate WID - Fbottom

# Calculate Xc and Xf
# 	Crest location < -Ld/2 --> Xc = -Ld/2
#	Crest location > -Ld/2 --> Xc = Crest location 
# 
#	Front of wave < Ld/2 --> Xf = Front of wave crossing with topsides BOS
#	Front of wave > Ld/2 --> Xf = Ld/2

# Calculate relative distance between crest and front/back of the wave (at Hd Level)
delta_t = t2 - t1
crest_to_foot = round((delta_t / T) * L, 3) 

# Calculate Xc and Xf for every time step
# Define starting point
loc_front = -Ld/2					# Location of front wave at time[0]
loc_crest = -Ld/2 - crest_to_foot	# Location of wave crest at time[0]

# length step 
length_step = (t6-t1)/100 / T * L

xc_xf = []	# [Xc, Xf]
for index in range(0, len(time_range)) :
	
	loc_front_i = loc_front + length_step * index
	loc_crest_i = loc_crest + length_step * index

	if loc_crest_i < -Ld/2 :
		Xc = -Ld/2

	elif loc_crest_i >= Ld/2 :
		Xc = Ld/2

	else :
		Xc = loc_crest_i


	if loc_front_i <= Ld/2 :
		Xf = loc_front_i

	else :
		Xf = Ld/2

	xc_xf.append([round(Xc, 3), round(Xf, 3)])

# print (*xc_xf,sep="\n")



#print (sample_range(xc_xf[2][0], xc_xf[2][1], 0.1 ))



# Calculate all data required to calculate FB based on time step

# Calculate integration points on each time step 
loc_list = []
for index in range(0, len(time_range)):

	if xc_xf[index][0] == xc_xf[index][1] :  # should put this inside function
		loc_list.append([0])

	else :
		loc_list.append(sample_range(xc_xf[index][0], xc_xf[index][1], 0.05))



#print (*loc_list, sep="\n")
# print("length of loc_list :", len(loc_list))
# print("length of time_range loop :", len(range(0, len(time_range))))
# print(len(time_range))


# Calculate horizontal particle velocities for each integration points
# for each time step

u_list = []
for index in range(0, len(time_range)):

	u_list_each=[]
	if len(loc_list[index]) == 1 :  # should put this inside function
		u_list_each = [0]

	else :
		for each_loc in loc_list[index] :
			u_list_each.append(wave_hor_vel(D, T, H, L, time_range[index], each_loc, 
								wave_surface(D, T, H, L, time_range[index], each_loc)  ))

	u_list.append(u_list_each)


# Calculate vertical particle velocities for each integration points
# for each time step

v_list = []
for index in range(0, len(time_range)):

	v_list_each=[]
	if len(loc_list[index]) == 1 :  # should put this inside function
		v_list_each = [0]

	else :
		for each_loc in loc_list[index] :
			v_list_each.append(wave_ver_vel(D, T, H, L, time_range[index], each_loc, 
								wave_surface(D, T, H, L, time_range[index], each_loc)  ))

	v_list.append(v_list_each)


# # Check length of every item in u_list
# for x in v_list :
# 	print (len(x))

# Multiply u and v for every point for every time step
uv_list = []
for i in zip(u_list, v_list):
	uv_list.append([u*v for u,v in zip(i[0], i[1])])

# Calculate FB
force_bottom_list = []
for item in zip(loc_list, uv_list):
	force_bottom_list.append( RHO * B * simps(item[1], item[0]) )



# print (*xc_xf,sep="\n")
# print(*loc_list, sep="\n")

'''
print(time_range[13])
print(xc_xf[13])
print(len(loc_list[13]))
print(len(u_list[13]))
print(len(v_list[13]))




# Try for only 1 time step
# Index no 13 is when the crest at the front of topsides
uv_single =[c*d for c, d in zip(u_list[13], v_list[13])]
print(*uv_single, sep="\n")

single_test = simps(uv_single, loc_list[13])
print(single_test)
'''



plt.figure(1)
plt.plot(time_range, force_front_list, time_range, force_bottom_list)
plt.show()


# # # # # plt.subplot(212)
# # # # # plt.plot(loc_lis[2], z_local)

# plt.show()
















