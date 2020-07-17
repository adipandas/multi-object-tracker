from ipyfilechooser import FileChooser


def create_filechooser(default_path="~/", html_title="Select File", use_dir_icons=True):
    fc = FileChooser(default_path)
    fc.title = html_title
    fc.use_dir_icons = use_dir_icons
    return fc


def select_caffemodel_prototxt(default_path="~/", use_dir_icons=True):
    html_title = '<b>Select <code>.prototxt</code> file for the caffemodel:</b>'
    fc = create_filechooser(default_path=default_path,
                            html_title=html_title,
                            use_dir_icons=use_dir_icons)
    return fc


def select_caffemodel_weights(default_path="~/", use_dir_icons=True):
    html_title = '<b>Select caffemodel weights (file with extention <code>.caffemodel</code>):</b>'
    fc = create_filechooser(default_path=default_path,
                            html_title=html_title,
                            use_dir_icons=use_dir_icons)
    return fc


def select_caffemodel(default_path="~/", use_dir_icons=True):
    prototxt = select_caffemodel_prototxt(default_path=default_path, use_dir_icons=use_dir_icons)
    weights = select_caffemodel_weights(default_path=default_path, use_dir_icons=use_dir_icons)
    return prototxt, weights


def select_videofile(default_path="~/", use_dir_icons=True):
    html_title = '<b>Select video file:</b>'
    fc = create_filechooser(default_path=default_path,
                            html_title=html_title,
                            use_dir_icons=use_dir_icons)
    return fc


def select_yolo_weights(default_path="~/", use_dir_icons=True):
    html_title = '<b>Select YOLO weights (<code>.weights</code> file):</b>'
    fc = create_filechooser(default_path=default_path,
                            html_title=html_title,
                            use_dir_icons=use_dir_icons)
    return fc


def select_coco_labels(default_path="~/", use_dir_icons=True):
    html_title = '<b>Select coco labels file (<code>.name</code> file):</b>'
    fc = create_filechooser(default_path=default_path,
                            html_title=html_title,
                            use_dir_icons=use_dir_icons)
    return fc


def select_yolo_config(default_path="~/", use_dir_icons=True):
    html_title = '<b>Choose YOLO config file (<code>.cfg</code> file):</b>'
    fc = create_filechooser(default_path=default_path,
                            html_title=html_title,
                            use_dir_icons=use_dir_icons)
    return fc


def select_yolo_model(default_path="~/", use_dir_icons=True):
    yolo_weights = select_yolo_weights(default_path, use_dir_icons)
    yolo_config = select_yolo_config(default_path, use_dir_icons)
    coco_names = select_coco_labels(default_path, use_dir_icons)
    return yolo_weights, yolo_config, coco_names


def select_pbtxt(default_path="~/", use_dir_icons=True):
    html_title = '<b>Select <code>.pbtxt</code> file:</b>'
    fc = create_filechooser(default_path=default_path,
                            html_title=html_title,
                            use_dir_icons=use_dir_icons)
    return fc


def select_tfmobilenet_weights(default_path="~/", use_dir_icons=True):
    html_title = '<b>Select tf-frozen graph of mobilenet (<code>.pb</code> file):</b>'
    fc = create_filechooser(default_path=default_path,
                            html_title=html_title,
                            use_dir_icons=use_dir_icons)
    return fc


def select_tfmobilenet(default_path="~/", use_dir_icons=True):
    prototxt = select_pbtxt(default_path, use_dir_icons)
    tfweights = select_tfmobilenet_weights(default_path, use_dir_icons)
    return prototxt, tfweights
