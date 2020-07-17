from ipyfilechooser import FileChooser


def create_filechooser(default_path="~/", html_title="Select File", use_dir_icons=True):
    fc = FileChooser(default_path)
    fc.title = html_title
    fc.use_dir_icons = use_dir_icons
    return fc


def select_caffemodel_prototxt(default_path="~/", use_dir_icons=True):
    html_title = '<b>Choose <code>.prototxt</code> file for the caffemodel:</b>'
    fc = create_filechooser(default_path=default_path,
                            html_title=html_title,
                            use_dir_icons=use_dir_icons)
    return fc


def select_caffemodel_weights(default_path="~/", use_dir_icons=True):
    html_title = '<b>Choose caffemodel weights (file with extention <code>.caffemodel</code>):</b>'
    fc = create_filechooser(default_path=default_path,
                            html_title=html_title,
                            use_dir_icons=use_dir_icons)
    return fc


def select_caffemodel(default_path="~/", use_dir_icons=True):
    prototxt = select_caffemodel_prototxt(default_path=default_path, use_dir_icons=use_dir_icons)
    weights = select_caffemodel_weights(default_path=default_path, use_dir_icons=use_dir_icons)
    return prototxt, weights


def select_videofile(default_path="~/", use_dir_icons=True):
    html_title = '<b>Choose video file:</b>'
    fc = create_filechooser(default_path=default_path,
                            html_title=html_title,
                            use_dir_icons=use_dir_icons)
    return fc
