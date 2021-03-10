# From process_files


self.coor = pd.read_csv("./Data83018/ROIcoords.csv")
self.coor_ordered = process_files.process_roi_coord(coor=self.coor, roi_names=self.ROInames)
def process_roi_coord(coor, roi_names):
    """ Function to process roi coordinates table. Reindex the coordinate table.
    ---
    Inputs:
        coor --> ROIcoords.csv table
        roi_names --> ROInames created in process_pathdata
    ---
    Outputs:
        coor --> input matrix with sorted rows
    """

    coor.rename(columns={'Unnamed: 0': 'ROI'}, inplace=True)
    idx = []
    for i in range(0, len(roi_names)):  # Reordering according to ROInames
        for k in range(0, len(coor['ROI'])):
            if roi_names[i] == coor['ROI'][k]:
                idx.append(k)
    coor = coor.loc[idx, :]
    # From the new index created we reorganize the table by index.

    return coor
