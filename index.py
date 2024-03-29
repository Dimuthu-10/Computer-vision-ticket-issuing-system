import base64
import os
from io import BytesIO as _BytesIO

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import tensorflow as tf
import dash_bootstrap_components as dbc
import mysql.connector
from mysql.connector import Error
from PIL import Image
from dash.dependencies import Input, Output, State
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

from app import app

app = dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    html.Nav([
    dbc.NavbarSimple(
    brand="Highway Web APP",
    brand_href="#",
    color="info",
    dark=True,),
    ]),
    html.Div(id='main-div', children=[
        html.Div(id='upload-div', children=[
            html.Div(id='upload-div-wrap', children=[
                html.Div(id='output-image-upload'),
                html.Div(id='upload',
                    children = dcc.Upload(
                        id='upload-image',
                        children=html.Div([
                            dbc.Button("Add files", color="primary", className="me-1")]),
                        multiple=False,
                    )),
            ]),
            html.Div(id='price-card-wrap',
                children=html.Div(
                    id='price-card',
                    children=dbc.Card(
                        [dbc.CardBody(
                           id='card-body'
                        )],
                        style={"width": "12rem", 'height': '12rem'},
                    )
            ))
        ]),
        #dropdowns
        html.Div(id='dropdown-wrap', children=[
            html.Div(id='entrance-wrap', children=[
                html.Div(id='entrance-label',children = html.H2('Entrance Location')),
                dcc.Dropdown(id='Entrance-locations',
                     options= [{'label' :'Ja-ela', 'value': '1'},
                            {'label' : 'Kerawalapitiya', 'value': '2'},
                            {'label' : 'Peliyagoda', 'value': '3'},
                            {'label' : 'Kadawatha', 'value': '4'},
                            {'label' : 'Kaduwela', 'value': '5'},
                            {'label' : 'Athurugiriya', 'value': '6'},
                            {'label' : 'Kottawa', 'value': '7'},
                            {'label' : 'Kahathuduwa', 'value': '8'},
                            {'label' : 'Gelanigama', 'value': '9'},
                            {'label' : 'Dodangoda', 'value': '10'},
                            {'label' : 'welipenna', 'value': '11'},
                            {'label' : 'Kurundugahahetepma', 'value': '12'},
                            {'label' : 'Baddeagama', 'value': '13'},
                            {'label' : 'Pinnaduwa', 'value': '14'},
                            {'label' : 'Imaduwa', 'value': '15'},
                            {'label' : 'Kokmaduwa', 'value': '16'},
                            {'label' : 'Godagama', 'value': '17'}],
                     value='1'
                ),
                html.Div(id='entrance-container')
            ]),
            html.Div(id='exit-wrap', children=[
                html.Div(id='exit-label',children = html.H2('Exit Location')),
                dcc.Dropdown(id='exit-locations',
                     options=[{'label': 'Ja-ela', 'value': '1'},
                              {'label': 'Kerawalapitiya', 'value': '2'},
                              {'label': 'Peliyagoda', 'value': '3'},
                              {'label': 'Kadawatha', 'value': '4'},
                              {'label': 'Kaduwela', 'value': '5'},
                              {'label': 'Athurugiriya', 'value': '6'},
                              {'label': 'Kottawa', 'value': '7'},
                              {'label': 'Kahathuduwa', 'value': '8'},
                              {'label': 'Gelanigama', 'value': '9'},
                              {'label': 'Dodangoda', 'value': '10'},
                              {'label': 'welipenna', 'value': '11'},
                              {'label': 'Kurundugahahetepma', 'value': '12'},
                              {'label': 'Baddeagama', 'value': '13'},
                              {'label': 'Pinnaduwa', 'value': '14'},
                              {'label': 'Imaduwa', 'value': '15'},
                              {'label': 'Kokmaduwa', 'value': '16'},
                              {'label': 'Godagama', 'value': '17'}],
                     value='1'),
                html.Div(id='exit-container')
            ]),

        ])
        #dropdown end
    ]),



    html.Div(id='results',
             style={
                 'width': '50%'
             })
], style={})

@app.callback([Output('output-image-upload', 'children'),
               Output('card-body', 'children')],
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'))
def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        return [[parse_contents(list_of_contents, list_of_names)], [parse_image(list_of_contents)]]

@app.callback(
    Output('entrance-container', 'children'),
    Input('Entrance-locations', 'value'))
def update_output(value):
    return f'You have selected {value}'

@app.callback(
    Output('exit-container', 'children'),
    Input('exit-locations', 'value'))
def update_output(value):
    return f'You have selected {value}'


#Functions

#functions related to database Connection
def database_connection():
    try:
        connection = mysql.connector.connect(host='localhost',
                                             database='highway',
                                             user='root',
                                             password='')
        if connection.is_connected():
            db_Info = connection.get_server_info()
            print("Connected to MySQL Server version ", db_Info)
            cursor = connection.cursor()
            cursor.execute("select database();")
            record = cursor.fetchone()
            print("You're connected to database: ", record)
    except Error as e:
        print("Error while connecting to MySQL", e)
    # finally:
    #     if connection.is_connected():
    #         cursor.close()
    #         connection.close()
    #         print("MySQL connection is closed")

def database_insert():
    try:
        connection = database_connection()

        # the query that need to enter
        insert_query = """INSERT INTO vehicle_records (Id, vehicle_number, vehicle type, vehicle_category_id,
                            entrance_location_id,entrance_dateTime,exit_location_id,exit_dateTime,price_id) 
                           VALUES 
                           (15, 'Lenovo ThinkPad P71', 6459, '2019-08-14') """

        cursor = connection.cursor()
        cursor.execute(insert_query)
        connection.commit()
        print(cursor.rowcount, "Record inserted successfully into Laptop table")
        cursor.close()
    except mysql.connector.Error as error:
        print("Failed to insert record into table {}".format(error))
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")

def database_read():
    try:
        connection = database_connection()

        #the query that need to enter
        select_Query = "select * from Laptop"
        cursor = connection.cursor()
        cursor.execute(select_Query)

    except mysql.connector.Error as e:
        print("Error reading data from MySQL table", e)
    finally:
        if connection.is_connected():
            connection.close()
            cursor.close()
            print("MySQL connection is closed")


#functions related to machine learning model
def b64_to_pil(string):
    decoded = base64.b64decode(string)
    buffer = _BytesIO(decoded)
    im = Image.open(buffer)
    rgb = Image.new('RGB', im.size)
    rgb.paste(im)
    image = rgb
    test_image = image.resize((128, 128))
    test_image1 = image.resize((224, 224))

    return test_image,test_image1


def b64_to_numpy(string, to_scalar=True):
    im, im1 = b64_to_pil(string)
    np_array = np.asarray(im)
    np_array1 = np.asarray(im1)

    if to_scalar:
        np_array = np_array / 255.
        np_array1 = np_array1 / 255.

    return np_array,np_array1


def parse_contents(contents, filename):
    return html.Div([
        # html.H5(filename),
        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents,
                 style={'width': '100%'})
    ], style={'width': '100%',
              'height': 'fit-content',
              'padding': '28px',
              'box-sizing': 'border-box'})


def parse_image(contents):
    classes_ = ['Ambulance', 'Bus', 'Car', 'Limousine', 'Motorcycle', 'Taxi', 'Truck', 'Van']
    classes1_ = ['not_vehicle', 'vehicle']

    model1 = tf.keras.models.load_model('models/Resnet50.hdf5')  # load the vehicle and non vehicle model
    content_type, content_string = contents.split(",")
    image,image1 = b64_to_numpy(content_string)
    original_img = b64_to_pil(content_string)[1]
    image = image.reshape((1, 128, 128, 3))
    image1 = image1.reshape(1, 224, 224, 3)
    pred1 = model1.predict(image)
    p1 = zip(list(classes1_), list(pred1[0]))
    p1 = sorted(list(p1), key=lambda z: z[1], reverse=True)[0][0]

    # print(p1[0][0])
    if p1 is 'vehicle':
        print(p1)
        model = tf.keras.models.load_model('models/googlenet.hdf5')
        pred = model.predict(image1)
        p = zip(list(classes_), list(pred[0]))
        p = sorted(list(p), key=lambda z: z[1], reverse=True)[:20]
        temp = pd.DataFrame(data=p, columns=['label', 'prob'])
        temp['text'] = [f'{round(t * 100, 2)}%' for t in temp.prob]

        numberplate = number_plate(original_img)

        # bar = go.Figure(data=[go.Bar(x=temp.prob,
        #                              y=temp.label,
        #                              text=temp.text,
        #                              orientation='h')])
        # bar.update_layout(hovermode=False,
        #                   paper_bgcolor='#fff',
        #                   plot_bgcolor='#fff',
        #                   height=800,
        #                   margin_pad=10,
        #                   xaxis=dict(showline=False,
        #                              showgrid=False,
        #                              showticklabels=False),
        #                   yaxis=dict(showline=False,
        #                              showgrid=False)
        #                   )

        return html.Div([
            html.H4(f"Vehicle = {temp['label'][0]}", className="card-title"),
            html.P(f"Number = {numberplate}",
                   className="card-text")
        ])

    if p1 is not 'vehicle':
        print('this is not a vehicle')
        print(p1)

        return html.Div([html.H2('Vehicle detection web APP'),html.H3('This is not a vehicle')])

#Numberplate Reading

def number_plate(img):
    # #Making paths
    CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
    PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
    PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
    TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
    LABEL_MAP_NAME = 'label_map.pbtxt'

    paths = {
        'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
        'SCRIPTS_PATH': os.path.join('Tensorflow', 'scripts'),
        'APIMODEL_PATH': os.path.join('Tensorflow', 'models'),
        'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace', 'annotations'),
        'IMAGE_PATH': os.path.join('Tensorflow', 'workspace', 'images'),
        'MODEL_PATH': os.path.join('Tensorflow', 'workspace', 'models'),
        'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace', 'pre-trained-models'),
        'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME),
        'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'export'),
        'TFJS_PATH': os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'tfjsexport'),
        'TFLITE_PATH': os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'tfliteexport'),
        'PROTOC_PATH': os.path.join('Tensorflow', 'protoc')
    }

    files = {
        'PIPELINE_CONFIG': os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'pipeline.config'),
        'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME),
        'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
    }

    # #Making paths
    for path in paths.values():
        if not os.path.exists(path):
            if os.name == 'posix':
                os.mkdir(path)
            if os.name == 'nt':
                os.mkdir(path)
    # if not os.path.exists(os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection')):
    #     git clone https://github.com/tensorflow/models {paths['APIMODEL_PATH']}
    # else:
    #     print("file already exist")


    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-11')).expect_partial()

    @tf.function
    def detect_fn(image):
        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)
        return detections

    category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
    # IMAGE_PATH = os.path.join(paths['IMAGE_PATH'], 'test', 'Cars431.png')

    # img = cv2.imread(IMAGE_PATH)
    image_np = np.array(img)
    # image_np = input_image

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'] + label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=5,
        min_score_thresh=.5)
    # agnostic_mode=False)

    # plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
    # plt.show()

    # Reading numberplate


    def box_filters(threshold):
        scores = list()
        for i in detections['detection_scores']:
            # print(i,type(i))

            if i>threshold:
                scores.append(i)
        return scores

    import easyocr
    detection_threshold = 0.1
    image = image_np_with_detections
    # scores = list(filter(lambda x:(x > detection_threshold), detections['detection_scores']))
    scores = box_filters(detection_threshold)
    # print(scores)

    boxes = detections['detection_boxes'][:len(scores)]
    # print(detections['detection_scores'])

    classes = detections['detection_classes'][:len(scores)]
    # print(detections['detection_classes'][:len(scores)])

    width = image.shape[1]
    height = image.shape[0]

    # ocr_result = None
    # Apply ROI filtering and OCR
    ocr_result = None
    for idx, box in enumerate(boxes):
        # print(box)
        roi = box * [height, width, height, width]
        # print(roi)
        region = image[int(roi[0]):int(roi[2]), int(roi[1]):int(roi[3])]
        reader = easyocr.Reader(['en'])
        ocr_result = reader.readtext(region)
        # print(ocr_result)
        # plt.imshow(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
        for result in ocr_result:
            # print(np.sum(np.subtract(result[0][2],result[0][1])))
            print(result[1])

    return [result[1] for result in ocr_result]
#end of the numberplate reading




if __name__ == '__main__':
    app.run_server(debug=True)
