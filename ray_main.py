'''this module contains several different functions for optical plotting 
and calculating intersections and angle of refracted rays'''

import numpy
import matplotlib.pyplot as pylab
from scipy.optimize import minimize
import scipy.optimize

def refraction_2d (incident_rays, planar_surface):
	'''this function returns the coordinates of intersection and refracted
	angle wrt vertical of incident rays intersecting with a planar surface'''
    
    # set up array to hold results
	refracted_rays = numpy.zeros(incident_rays.shape, dtype=float)
    
    # calculate critical angle
	if planar_surface[5] >= planar_surface[4]:
		# total internal reflection is not possible
		tir_possible = False
	else:
		tir_possible = True
		critical_angle = numpy.arcsin(planar_surface[5]/planar_surface[4])
	    
    # calculate angles to surface normal of incident rays
	planar_angle = numpy.arctan((planar_surface[2]-planar_surface[0])/(planar_surface[3]-planar_surface[1]))
	incident_angles =  incident_rays[:,2] - (numpy.pi/2.0) - planar_angle # transform ray angles to be with respect to normal to surface
        	
	# handle incident rays exceeding the critical angle
	if tir_possible and(abs(incident_angles) > critical_angle).any():
		raise Exception, "at least one incident ray exceeds the critical angle"
            
    # calculate gradients and intercepts of incoming rays and surface
	ray_gradients = numpy.tan((numpy.pi/2.0) - incident_rays[:,2])
	ray_intercepts = incident_rays[:,1] - (ray_gradients * incident_rays[:, 0])
	surface_gradient = (planar_surface[3]-planar_surface[1])/(planar_surface[2]-planar_surface[0])
	surface_intercept = planar_surface[1] - (surface_gradient * planar_surface[0])

    # calculate points of intersection of rays with surface...
    # horizontal
	refracted_rays[:,0] = (surface_intercept - ray_intercepts) / (ray_gradients - surface_gradient)
    # vertical
	refracted_rays[:,1] = (ray_gradients * refracted_rays[:,0]) + ray_intercepts
    
    # calculate directions of refracted rays
	refracted_angles = numpy.arcsin(planar_surface[4]*numpy.sin(incident_angles)/planar_surface[5]) # Snell's law
	refracted_rays[:,2] = refracted_angles + (numpy.pi/2.0) + planar_angle # transform output ray angles to be clockwise from vertical
    
	return refracted_rays

def refraction_2d_sph (incident_rays, spherical_surface):
	'''this function returns the coordinates of intersection 
	and refracted angle wrt vertical of incident rays 
	intersecting with a spherical surface'''
	
	xor = incident_rays[:,0]			#x coordinate of origin of incident ray
	yor = incident_rays[:,1]			#y coordinate of origin of incident ray
	rayang = incident_rays[:,2]			#clockwise angle with respect to verticle of incident ray
	xlens = spherical_surface[0]		#x coordinate of the upper end of the arc
	ylens = spherical_surface[1]		#y coordinate of the upper end of the arc
	ninc = spherical_surface[4]			#refractive index of the incident side
	nref = spherical_surface[5]			#refractive index of the refracted side
	rad = spherical_surface[6]			#radius of curvature
	
	if (nref >= ninc):
		totintref_possible = False		#total internal reflection is possible if n2 > n1
	else:
		totintref_possible = True
		critical_angle = numpy.arcsin(nref/ninc)	#critical angle for material
	
	l = (rad**2.0 - ylens**2.0)**0.5	#distance between x coordinates of arc and circle centre
	
	if (rad > 0.0):
		xciror = xlens + l				#x coordinate for circle centre
	else:
		xciror = xlens - l				#x coordinate for circle centre
	
	m = numpy.tan(numpy.pi/2.0-rayang)	#grandient of incident light ray
	c = yor - m*xor						#y intercept of the incident light ray
	aroot = (m**2.0 + 1.0)						#2nd order term of quadratic equation to find x intercept
	broot = (2.0*m*c - 2.0*xciror)				#1st order term of quadratic equation to find x intercept
	croot = (c**2.0 + xciror**2.0 - rad**2.0)	#0th order term of quadratic equation to find x intercept
	if (rad > 0.0):
		xint = ( -broot - (broot**2.0 - 4.0*aroot*croot)**0.5)/(2.0*aroot)	#x coordinate of incident ray intersecting with arc
	else:
		xint = ( -broot + (broot**2.0 - 4.0*aroot*croot)**0.5)/(2.0*aroot)	#x coordinate of incident ray intersecting with arc
	yint = m*xint + c							#y coordinate of incidence ray intersection with arc
	
	mn = (yint)/(xint - xciror)					#gradient of circle at point of intersection with incident ray
	angn = numpy.arctan(mn)						#angle of normal to the horizontal
	anginc = numpy.pi/2.0 - rayang - angn		#angle of incidence of incident ray
	angrefra = numpy.arcsin(ninc*numpy.sin(anginc)/nref)		#angle of refraction of the refracted ray
	angrefravert = numpy.pi/2.0 - angrefra - angn				#refracted angle with respect to the verticle
	
	#error message if total internal reflection occurs
	if totintref_possible and(abs(anginc) > critical_angle).any():
		raise Exception, "at least one incident ray exceeds the critical angle"

	
	refracted_rays = numpy.array([xint, yint, angrefravert])	#array containing results
	return numpy.transpose(refracted_rays)

def refraction_2d_det (incident_rays, x_det):
	'''this function returns the coordinates of 
	intersection and refracted angle wrt vertical 
	of incident rays intersecting with a detector surface'''

	xor = incident_rays[:,0]					#x coordinate of origin of incident ray
	yor = incident_rays[:,1]					#y coordinate of origin of incident ray
	rayang = incident_rays[:,2]					#clockwise angle with respect to verticle of incident ray
	m = numpy.tan(numpy.pi/2.0 - rayang)		#gradient of incident ray
	c = yor - m*xor								#y intercept of incident ray
	yint = m*x_det + c							#y coordinate of intersection
	
	refracted_rays = numpy.zeros(incident_rays.shape, dtype=float)		#setup array to hold results
	refracted_rays[:,0] = x_det
	refracted_rays[:,1] = yint
	refracted_rays[:,2] = 0.0

	return refracted_rays

def trace_2d (incident_rays, surface_list):
	'''this function returns an array of coordinates
	of intersection and refracted angle wrt vertical 
	of incident rays intersecting with a number of 
	different surfaces enumerately giving each set 
	of results for each surface'''
	
	#setup array to contain results
	refracted_ray_paths = numpy.zeros([len(surface_list), incident_rays.shape[0], incident_rays.shape[1]])
		
	for i in range(len(surface_list)):					#to repeat over all of the surfaces
		
		if surface_list[i][0] == 'DET':					#where indexed value is indicative of surface type
			x_det = surface_list[i][1]					#indexing domain of function required
			#calling function to calculate results
			refracted_ray_paths[i] = refraction_2d_det(incident_rays, x_det)
			
		elif surface_list[i][0] == 'PLA':				#where indexed value is indicative of surface type
			planar_surface = surface_list[i][1]			#indexing domain of function required
			#calling function to calculate results
			refracted_ray_paths[i] = refraction_2d(incident_rays, planar_surface)
			
		elif surface_list[i][0] == 'SPH':				#where indexed value is indicative of surface type
			spherical_surface = surface_list[i][1]		#indexing domain of function required
			#calling function to calculate results
			refracted_ray_paths[i] = refraction_2d_sph(incident_rays, spherical_surface)
		
		incident_rays = refracted_ray_paths[i]			#replacing domains of functions with range values to find continuing results of ray
		
	return refracted_ray_paths
	
def plot_trace_2d (incident_rays, refracted_ray_paths, surface_list):
	'''this function produces a plot of the rays 
	and surfaces specified in the function trace_2d, 
	also a plot of the 1D image on the final surface 
	is produced '''
	
	f, (trace,det) = pylab.subplots(1,2, sharey=False)	#setup subplot of trace and 1D image of rays on detector
		
	ymaxs = []											#setup list to find largest y coordinates of any surface - done to setup appropriate axis of plot
	ymins = []											#setup list to find largest y coordinates of any surface - done to setup appropriate axis of plot
	
	xs = refracted_ray_paths[:,:,0] 					#array of all x coordinates of intersection with all surfaces
	ys = refracted_ray_paths[:,:,1]						#array of all y coordinates of intersection with all surfaces
		
	for n in range(incident_rays.shape[0]):				#repeats over all incident rays
		ymaxs.append(incident_rays[n,1])				#adds y coordinate of incident ray to list for plotting purposes
		ymins.append(incident_rays[n,1])				#adds y coordinate of incident ray to list for plotting purposes

		for i in range(len(xs) - 1):					#repeats of all refracted rays
			x1 = xs[i,n]								#x coordinate of origin of refracted rays
			x2 = xs[i + 1,n]							#x coordinate of intersection of refracted rays
			y1 = ys[i,n]								#y coordinate of origin of refracted rays
			y2 = ys[i + 1,n]							#y coordinate of intersection of refracted rays
			trace.plot((x1,x2),(y1,y2), color='blue')	#plotting rays between coordinates just specified
		#plotting the initial incident rays
		trace.plot((incident_rays[n,0],xs[0,n]), (incident_rays[n,1],ys[0,n]), color='blue')

	for i in range(len(surface_list)):					#repeat over all surfaces
	
		type = surface_list[i][0]						#extracts type of surface eg PLA or DET or SPH
		surface = surface_list[i][1]					#getting the property values for each optical element
			
		if type == 'PLA':
		
			ymaxs.append(surface[3])					#add y coordinates of surface to list
			ymaxs.append(surface[1])					#add y coordinates of surface to list
			ymins.append(surface[3])					#add y coordinates of surface to list
			ymins.append(surface[1])					#add y coordinates of surface to list
		
			#calculate gradient of planar surface
			surface_gradient = (surface[3]-surface[1])/(surface[2]-surface[0])
			#calculate y-intercept of planar surface
			surface_intercept = surface[1] - (surface_gradient*surface[0])	
			
			#for plotting surfaces
			xpvals = numpy.linspace(surface[0],surface[2],50)			#x values over which surface should be plotted
			ypvals = surface_gradient*xpvals + surface_intercept		#y values over which surface should be plotted
			trace.plot(xpvals,ypvals, color='black')
		
		elif type == 'SPH':
		
			ymaxs.append(surface[3])					#add y coordinates of surface to list
			ymaxs.append(surface[1])					#add y coordinates of surface to list
			ymins.append(surface[3])					#add y coordinates of surface to list
			ymins.append(surface[1])					#add y coordinates of surface to list
			
			xlens = surface[0]							#x coordinate of an end of the arc
			ylens = surface[1]							#y coordinate of an end of the arc
			rad = surface[6]							#radius of curvature
						
			l = (rad**2.0 - ylens**2.0)**0.5			#distance between x coordinates of an end of the arc and circle centre
			
			if (rad > 0.0):
				xciror = xlens + l									#x coordinate for circle centre
				butress = xciror - rad								#x coordinate along the y = 0 axis
				xsvals = numpy.linspace(butress+1e-6,xlens,100)		#x values over which the surface should be plotted
			else:
				xciror = xlens - l									#x coordinate for circle centre
				butress = xciror - rad								#x coordinate along the y = 0 axis
				xsvals = numpy.linspace(xlens,butress-1e-6,100)		#x values over which the surface should be plotted
			
			ysvals = ((rad)**2 - (xsvals - xciror)**2)**0.5			#y values over which the surface should be plotted#arc
			trace.plot(xsvals,ysvals, color='black')				#plotting the northern hemiarc
			trace.plot(xsvals,-ysvals, color='black')				#plotting the southern hemiarc
			
		
		elif type == 'DET':
			
			xdet = surface											#x coordinate of detector

			yrmax = numpy.amax(refracted_ray_paths[i,:,1])						#finding the highest y intersection on detector - done for plotting of detector purposes
			yrmin = numpy.amin(refracted_ray_paths[i,:,1])						#finding the lowest y intersection on detector - done for plotting of detector purposes
			trace.plot((xdet, xdet), (yrmin - 1, yrmax + 1), color='black')		#plotting detector just above and below highest and lowest y intersection
			ymaxs.append(yrmax)													#adding max ray intercepts to list
			ymins.append(yrmin)													#adding min ray intercepts to lis
			ymax = numpy.amax(ymaxs)									#highest y coordinate for any surface
			ymin = numpy.amin(ymins)									#lowest y coordinate for any surface
			
			trace.axis([0.0, xdet + 1.0, ymin - 2.0, ymax + 2.0])		#setting axis of trace plot appropriatly
			
			yints = refracted_ray_paths[i,:,1]							#array of all y intersectings on detector
			det.plot(numpy.zeros_like(yints),yints, 'x')				#plotting 1D image of all rays on detector
			det.axis([-1.0, 1.0, yrmin - 1.0, yrmax + 1.0])				#setting axis of 1D image appropriatly
			
	trace.set_title('Trace plot')						#adding title
	trace.set_xlabel('Horizontal axis (meters)')		#adding axis title
	trace.set_ylabel('Vertical axis (meters)')			#adding axis title
	det.set_title('Detector image')						#adding title
	det.set_xlabel('Detector face')						#adding axis title
	det.axes.get_xaxis().set_visible(False)				#removing axis numbers
	det.set_ylabel('Vertical axis (meters)')			#adding axis title
	pylab.show()
	
def evaluate_trace_2d(refracted_ray_paths, r):
	'''this function returns the fraction of 
	rays arriving at the final surface that 
	fall within the 'circle' specifiedby a 
	given radius r. The centre of this circle
	is the mean vertical postion of all arriving rays'''
	
	#indexing to get array of all y coordinates of intersection with detector for 1D image plot
	detyints = refracted_ray_paths[:,:,1][len(refracted_ray_paths) - 1,:]
	meandetyints = numpy.mean(detyints)				#mean average of all y coordinates of intersection with detector
	lower = meandetyints - r						#lower y coordinate of circle radius r with centre at the mean defined before
	upper = meandetyints + r						#upper y coordinate of circle radius r with centre at the mean defined before
	totrayno = len(detyints)						#total number of incident rays
	hit = 0											#to start of number of ray hits inside circle radius r
	for i in range(totrayno):						#repeats over all final rays
		if lower < detyints[i] < upper:				#if ray lands inside cirlce a hit is counted
			hit = hit + 1							#counting number of hits of rays inside cirlce radius r
	frac = float(hit)/totrayno						#fraction of number of hits as a total of all final rays
	return frac
	
def optimize_surf_rad_2d(incident_rays, surface_list, r, n_surf):
	'''this function returns a value of the optimal 
	radius of curvature for a given spherical 
	surface in order to maximize the fraction 
	of rays that land withing the circle specified 
	in the function evaluate_trace_2d'''
	radii = numpy.zeros(len(n_surf), dtype=float)		#setup array to hold results
	for i,n in enumerate(n_surf):						#repeats one by one
		radii[i] = surface_list[n][1][6]				#indexing and defining radii to be optimized from surface_list

	def swap(radii):									#function to swap radii
		for i,n in enumerate(n_surf):					#repeat one by one
			surface_list[n][1][6] = radii[i]			#swapping back radii into list
			#calling trace_2d to do its magic
			refracted_ray_paths = trace_2d(incident_rays,surface_list)
			#calling evaluate_trace_2d to do its magic
			frac = evaluate_trace_2d(refracted_ray_paths,r)
		return (-1.0)*frac								#need negative of fraction for optimization

	#getting the optimal radius of curvature to maximize frac
	rad_opt = scipy.optimize.minimize(swap,radii,method='Nelder-Mead',tol=1e-6)
	return rad_opt.x
