import numpy as np# importing csv module
import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sc
import scipy.interpolate as interp

def extractData(filename):
    # csv file name
    
    
    # initializing the titles and rows list
    fields = []
    rows = []
    
    # reading csv file
    with open(filename, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile,delimiter=';')
        
        # extracting field names through first row
        fields = next(csvreader)
    
        # extracting each data row one by one
        for row in csvreader:
            rows.append(row)
        
        # get total number of rows
        print("Total no. of rows: %d"%(csvreader.line_num))
    
    # printing the field names
    print('Field names are:' + ', '.join(field for field in fields))

    #Extract data   
    temp = np.array([])
    data = np.empty((1,8))
    for row in rows:
        for col in row:
            tempVal = float(col)
            temp = np.append(temp,[tempVal])
            # print(temp.shape)
        data = np.vstack((data,temp.reshape(1,8)))
        # print(data)
        temp = np.array([])
    data = np.delete(data,0,axis = 0)
    print(f'The imported data size is {data.shape}')
    return data[1:,:]

def interpTime(data,dt):

    #interpolate 
    # TODO: Edit the column number index

    # time = data[:,0]
    #Partition data into 1D vectors
    time = data[:,0]
    x = data[:,2]
    y = data[:,3]
    # print(y)
    psi = data[:,4]
    v = data[:,6]

    # #test
    # time = [t for t in range(len(data))]


    #Interpolation grid
    tGrid = np.arange(0,time[-1]+dt,dt)
    # print(tGrid)
    tGrid = np.reshape(tGrid,(len(tGrid),1)) 
    
    # #1D Spline Interpolation
    x_interp_fcn = interp.splrep(time,x)
    # print(x_interp_fcn)
    y_interp_fcn = interp.splrep(time,y)
    psi_interp_fcn = interp.splrep(time,psi)
    v_interp_fcn = interp.splrep(time,v)

    #Interpolation cont...
    x_interp = interp.splev(tGrid, x_interp_fcn, der=0).reshape(len(tGrid),1)
    y_interp = interp.splev(tGrid, y_interp_fcn, der=0).reshape(len(tGrid),1)
    psi_interp = interp.splev(tGrid, psi_interp_fcn, der=0).reshape(len(tGrid),1)
    v_interp = interp.splev(tGrid, v_interp_fcn, der=0).reshape(len(tGrid),1)


    # #test interpolation
    # x_interp_fcn = interp.UnivariateSpline(time,x)
    # x_interp = x_interp_fcn(tGrid)
    # y_interp_fcn = interp.UnivariateSpline(time,y)
    # y_interp = y_interp_fcn(tGrid)
    # v_interp_fcn = interp.UnivariateSpline(time,v)
    # v_interp = v_interp_fcn(tGrid)
    # psi_interp_fcn = interp.UnivariateSpline(time,psi)
    # psi_interp = psi_interp_fcn(tGrid)


    # Export
    newData = np.hstack((tGrid,x_interp,y_interp,psi_interp,v_interp))
    return newData

 

if __name__ == '__main__':
    filename = '/home/han98122/Repositories/C231A/inputs/traj_race_cl_2.csv'
    data = extractData(filename)
    newData = interpTime(data,0.2)

    #csv export
    np.savetxt('/home/han98122/Repositories/C231A/inputs/test.csv', newData, delimiter=';')
    # with open('/home/han98122/Repositories/C231A/inputs/test.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     for row in newData
    #         writer.writerow(row) 
 