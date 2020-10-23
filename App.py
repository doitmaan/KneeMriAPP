import dash
import dash_html_components as html
import dash_core_components as dcc
import socket  

from plotly.tools import mpl_to_plotly
from plotly.subplots import make_subplots


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook
from tqdm.notebook import tqdm


import plotly.express as px

import plotly.graph_objects as go


import io
import base64


##data source 


train_acl = pd.read_csv('./MRNet/train-acl.csv', header=None,
                       names=['Case', 'Abnormal'], 
                       dtype={'Case': str, 'Abnormal': np.int64})

# print(train_acl.head())




case = '0000'

mri_coronal = np.load('MRNet/train/coronal/0000.npy')  
mri_axial = np.load('MRNet/train/axial/0000.npy')
mri_sagittal = np.load('MRNet/train/sagittal/0000.npy')

print(f'MRI scan on coronal plane: {mri_coronal.shape}')
print(f'MRI scan on axial plane: {mri_axial.shape}')
print(f'MRI scan on sagittal plane: {mri_sagittal.shape}')




# print("type is")
# print(type(mri_coronal[0, :, :]))
# fig(1,1) = px.imshow(mri_coronal[0, :, :])

# fig.add_image(go.Image(z=(mri_coronal[0, :, :])))
# fig.show()
# px.imshow


fig = px.imshow(mri_coronal[0, :, :])
fig1 = px.imshow(mri_axial[0, :, :])
fig2 = px.imshow(mri_sagittal[0, :, :])



# fig.update_layout(height=600, width=800, title_text="Side By Side Subplots")
# fig.show()

# ax1.imshow(mri_coronal[0, :, :], 'gray');

# ax1.set_title('Case 0 | Slice 1 | Sagittal');

# ax2.imshow(mri_axial[0, :, :], 'gray');
# ax2.set_title('Case 0 | Slice 1 | Axial');

# ax3.imshow(mri_sagittal[0, :, :], 'gray');
# ax3.set_title('Case 0 | Slice 1 | Coronal');





# plt.show()

# plotly_fig = mpl_to_plotly(fig)



           

train_path = 'MRNet/train/'

def load_one_stack(case, data_path=train_path, plane='coronal'):
    fpath = '{}/{}/{}.npy'.format(data_path, plane, case)
    return np.load(fpath)

def load_stacks(case, data_path=train_path):
    x = {}
    planes = ['coronal', 'sagittal', 'axial']
    for i, plane in enumerate(planes):
        x[plane] = load_one_stack(case, plane=plane)
    return x

def load_cases(train=True, n=None):
    assert (type(n) == int) and (n < 1250)
    if train:
        case_list = pd.read_csv('MRNet/train-acl.csv', names=['case', 'label'], header=None,
                               dtype={'case': str, 'label': np.int64})['case'].tolist()        
    else:
        case_list = pd.read_csv('MRNet/valid-acl.csv', names=['case', 'label'], header=None,
                               dtype={'case': str, 'label': np.int64})['case'].tolist()        
    cases = {}
    
    if n is not None:
        case_list = case_list[:n]
        
    for case in tqdm_notebook(case_list, leave=False):
        x = load_stacks(case)
        cases[case] = x
    return cases

cases = load_cases(n=100)

print(cases['0000'].keys())

print(cases['0000']['axial'].shape)
print(cases['0000']['coronal'].shape)
print(cases['0000']['sagittal'].shape)




class KneePlot():

    x = 0
    y = 0 
    z = 0
    
    def __init__(self, cases, figsize=(15, 5)):
        self.cases = cases
        
        self.planes = {case: ['coronal', 'sagittal', 'axial'] for case in self.cases}
    
        self.slice_nums = {}
        for case in self.cases:
            self.slice_nums[case] = {}
            for plane in ['coronal', 'sagittal', 'axial']:
                self.slice_nums[case][plane] = self.cases[case][plane].shape[0]

        self.figsize = figsize
        
    def _plot_slices(self, case, im_slice_coronal, im_slice_sagittal, im_slice_axial):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=self.figsize)
        
        ax1.imshow(self.cases[case]['coronal'][im_slice_coronal, :, :], 'gray')
        ax1.set_title(f'MRI slice {im_slice_coronal} on coronal plane')
        
        ax2.imshow(self.cases[case]['sagittal'][im_slice_sagittal, :, :], 'gray')
        ax2.set_title(f'MRI slice {im_slice_sagittal} on sagittal plane')
        
        ax3.imshow(self.cases[case]['axial'][im_slice_axial, :, :], 'gray')
        ax3.set_title(f'MRI slice {im_slice_axial} on axial plane')
        
        # plt.show()
        # return fig
    
    def draw(self):
        # case_widget = Dropdown(options=list(self.cases.keys()),
        #                        description='Case'
                              
        #                       )
        case_init = list(self.cases.keys())[0]


        slice_init_coronal = self.slice_nums[case_init]['coronal'] - 1    
        self.x =  slice_init_coronal   
        # slices_widget_coronal = IntSlider(min=0, 
        #                                   max=slice_init_coronal, 
        #                                   value=slice_init_coronal // 2, 
        #                                   description='Coronal')
        
        slice_init_sagittal = self.slice_nums[case_init]['sagittal'] - 1   
        self.y = slice_init_sagittal  
        # slices_widget_sagittal = IntSlider(min=0,
        #                                    max=slice_init_sagittal,
        #                                    value=slice_init_sagittal // 2,
        #                                    description='Sagittal'
        #                                   )
        
        slice_init_axial = self.slice_nums[case_init]['axial'] - 1   
        self.z = slice_init_axial 
        # print(z)
        # slices_widget_axial = IntSlider(min=0,
        #                                 max=slice_init_axial,
        #                                 value=slice_init_axial // 2,
        #                                 description='Axial'
        #                                )
        
        def update_slices_widget(*args):
            slices_widget_coronal.max = self.slice_nums[case_widget.value]['coronal'] - 1
            slices_widget_coronal.value = slices_widget_coronal.max // 2
            
            slices_widget_sagittal.max = self.slice_nums[case_widget.value]['sagittal'] - 1
            slices_widget_sagittal.value = slices_widget_sagittal.max // 2
            
            slices_widget_axial.max = self.slice_nums[case_widget.value]['axial'] - 1
            slices_widget_axial.value = slices_widget_axial.max // 2
    
        
        # case_widget.observe(update_slices_widget, 'value')
        # interact(self._plot_slices,
        #          case=case_widget, 
        #          im_slice_coronal=slices_widget_coronal, 
        #          im_slice_sagittal=slices_widget_sagittal, 
        #          im_slice_axial=slices_widget_axial
        #         )
    
    def resize(self, figsize): 
        self.figsize = figsize





obj = KneePlot(cases)
obj.draw()
print(list(obj.cases.keys()))



##APP

geberatedOptions = [{'label': i, 'value': i} for i in list(obj.cases.keys()) ]

print(geberatedOptions)





external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


# df = pd.DataFrame({
#     "x": [1,2,1,2],
#     "y": [1,2,3,4],
#     "customdata": [1,2,3,4],
#     "fruit": ["apple", "apple", "orange", "orange"]
# })

# fig = px.scatter(df, x="x", y="y", color="fruit", custom_data=["customdata"])

# fig.update_layout(clickmode='event+select')

# fig.update_traces(marker_size=20)



app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div([
    dcc.Slider(
        id='my-slider',
        min=0,
        max=obj.x,
        step=1,
        value=1, # here to be added given the size of the input 
    ),
    dcc.Slider(
        id='my-slider2',
        min=0,
        max=obj.y,
        step=1,
        value=1, # here to be added given the size of the input 
    ),
    dcc.Slider(
        id='my-slider3',
        min=0,
        max=obj.z,
        step=1,
        value=1, # here to be added given the size of the input 
    ),
    html.Div(id='slider-output-container'),
    html.Div(id='slider-output-container2'),
    html.Div(id='slider-output-container3'), 
   

    dcc.Dropdown(
        id='demo-dropdown',
        options=geberatedOptions
            # list(obj.cases.keys())
,
        value='NYC'
    ),
    html.Div(id='dd-output-container'),

    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '95%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        }, 
        # Allow multiple files to be uploaded
        multiple=True
    ),
     dcc.Graph(figure=fig , id = 'fig'),
     dcc.Graph(figure=fig1 , id = 'fig2'),
     dcc.Graph(figure=fig2, id = 'fig3')
  
])


@app.callback(
    dash.dependencies.Output('slider-output-container', 'children'),
    [dash.dependencies.Input('my-slider', 'value')])
def update_output(value):
    return 'You have selected case "{}"'.format(value)

@app.callback(
    dash.dependencies.Output('slider-output-container2', 'children'),
    [dash.dependencies.Input('my-slider2', 'value')])
def update_output(value):
    return 'You have selected "{}"'.format(value)   

@app.callback(
    dash.dependencies.Output('slider-output-container3', 'children'),
    [dash.dependencies.Input('my-slider3', 'value')])       
def update_output(value):
    return 'You have selected "{}"'.format(value)


@app.callback(
    dash.dependencies.Output('dd-output-container', 'children'),
    [dash.dependencies.Input('demo-dropdown', 'value')])
def update_output(value):
    return 'You have selected "{}"'.format(value)


@app.callback(
    dash.dependencies.Output('fig', 'figure'),
    [dash.dependencies.Input('my-slider', 'value')])
def update_fig(value):
    
    return px.imshow(mri_coronal[value, :, :]) 

@app.callback(
    dash.dependencies.Output('fig2', 'figure'),
    [dash.dependencies.Input('my-slider2', 'value')])
def update_fig(value):
    
    return px.imshow(mri_axial[value, :, :])    

@app.callback(
    dash.dependencies.Output('fig3', 'figure'),
    [dash.dependencies.Input('my-slider3', 'value')])
def update_fig(value):
    
    return px.imshow(mri_sagittal[value, :, :])       

   


if __name__ == '__main__':
    app.run_server( host = '127.0.0.1')
