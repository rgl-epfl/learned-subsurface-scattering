

import numpy as np
import pathlib

from nanogui import (Label, TextBox, Slider, Widget, BoxLayout, Orientation, GroupLayout,
                     Alignment, CheckBox, VScrollPanel, Button, Popup, PopupButton)


class ListPanel(Widget):
    def __init__(self, parent, items):
        super().__init__(parent)
        self.item_width = 400
        self.setFixedSize(np.array([self.item_width + 20, 260]))
        self.setLayout(BoxLayout(Orientation.Vertical,
                                 Alignment.Middle, 0, 0))
        self.cb = None
        self.selected = None
        self.vscroll = None
        self.setItems(items, np.arange(len(items)))
            
    def setItems(self, items, item_ids):
        if self.vscroll is not None:
            self.removeChild(self.vscroll)
        self.vscroll = VScrollPanel(self)
        self.vscroll.setFixedHeight(200)
        self.vscroll.setFixedWidth(self.item_width)
        self.itemwidget = Widget(self.vscroll)
        self.itemwidget.setLayout(BoxLayout(Orientation.Vertical,
                                            Alignment.Middle, 0, 0))

        def label_click_handler(button):
            self.selected = int(button.id())
            if self.cb is not None:
                self.cb(self.selected)

        self.list_items = []
        for i, item in enumerate(items):
            list_item = Button(self.itemwidget, item)
            list_item.setFixedWidth(self.item_width)
            list_item.setCallback(lambda button=list_item: label_click_handler(button))
            list_item.setId(str(item_ids[i]))
            list_item.setTooltip(item)
            list_item.setFontSize(14)
            list_item.setFlags(Button.Flags.RadioButton)
            self.list_items.append(list_item)
        self.selected = None
        self.setSelectedIndex(0)

    def setCallback(self, cb):
        self.cb = cb

    def setSelectedIndex(self, index):
        self.selected = index


class FilteredListPanel(ListPanel):
    def __init__(self, parent, items, parent_window):
        super().__init__(parent, items)
        self.parent_window = parent_window
        self.textBox = TextBox(self)
        self.removeChild(self.textBox)
        self.addChild(0, self.textBox)
        self.textBox.setFixedSize((self.item_width, 25))
        self.textBox.setEditable(True)
        self.textBox.setValue('*')
        self.textBox.setFontSize(20)
        self.textBox.setAlignment(TextBox.Alignment.Right)

        def cb_text(value):
            used_items = []
            indices = []
            for i, item in enumerate(self.items):
                if pathlib.PurePath(item.lower()).match(value.lower()):
                    used_items.append(item)
                    indices.append(i)
            self.setItems(used_items, indices)
            self.parent_window.performLayout()
            return True
        self.textBox.setCallback(cb_text)
        self.items = items


class FilteredPopupListPanel(Widget):

    def __init__(self, parent, items, parent_window, icon=None):
        super().__init__(parent)

        self.popupBtn = PopupButton(parent, items[0])
        popup = self.popupBtn.popup()
        popup.setLayout(GroupLayout())

        self.list_panel = FilteredListPanel(popup, items, parent_window)
        self.list_panel.setSelectedIndex(0)

    def setSelectedIndex(self, i):
        self.popupBtn.setCaption(self.list_panel.items[i])
        self.list_panel.setSelectedIndex(i)

    def setCallback(self, cb):
        def cb_wrapper(value):
            self.popupBtn.setCaption(self.list_panel.items[value])
            cb(value)
        self.list_panel.setCallback(cb_wrapper)


class LabeledSlider(Widget):

    def __init__(self, variable_scope, widget, variable_name, min_val, max_val,
                 dtype, callback=None, logspace=False, slider_width=120,
                 warp_fun=None, inv_warp_fun=None):
        super().__init__(widget)

        if logspace:
            warp_fun = np.log
            inv_warp_fun = np.exp

        if warp_fun is None:
            def warp_fun(x): return x

            def inv_warp_fun(x): return x

        self.warp_fun = warp_fun
        self.inv_warp_fun = inv_warp_fun

        self.variable_scope = variable_scope
        self.variable_name = variable_name
        self.logspace = logspace
        self.min_val = min_val
        self.max_val = max_val
        self.dtype = dtype

        Label(widget, variable_name)
        widget = Widget(widget)
        widget.setLayout(BoxLayout(Orientation.Horizontal,
                                   Alignment.Middle, 0, 20))

        self.slider = Slider(widget)

        diff = warp_fun(max_val) - warp_fun(min_val)
        self.slider.setValue((warp_fun(variable_scope.__dict__[variable_name]) - warp_fun(min_val)) / diff)
        self.slider.setFixedWidth(slider_width)

        self.textBox = TextBox(widget)
        self.textBox.setFixedSize((60, 25))
        self.textBox.setEditable(True)

        if dtype == int:
            self.textBox.setValue('{:d}'.format(variable_scope.__dict__[variable_name]))
        else:
            self.textBox.setValue('{:.3f}'.format(variable_scope.__dict__[variable_name]))

        self.textBox.setFontSize(20)
        self.textBox.setAlignment(TextBox.Alignment.Right)

        def cb(value):
            try:
                value = dtype(value)
            except ValueError:
                return False

            if value < min_val or value > max_val:
                return False

            self.slider.setValue(float((warp_fun(value) - warp_fun(min_val)) / diff))
            variable_scope.__dict__[variable_name] = value
            if callback:
                callback()
            return True

        self.textBox.setCallback(cb)

        def cb(value):
            format_str = {int: r'{:d}', float: r'{:.3f}'}
            self.textBox.setValue(format_str[dtype].format(dtype(inv_warp_fun(diff * value + warp_fun(min_val)))))

        self.slider.setCallback(cb)

        def cb(value):
            variable_scope.__dict__[variable_name] = dtype(inv_warp_fun(diff * value + warp_fun(min_val)))
            if callback:
                callback()
        self.slider.setFinalCallback(cb)

    def set_value(self, value):
        self.variable_scope.__dict__[self.variable_name] = value
        diff = self.warp_fun(self.max_val) - self.warp_fun(self.min_val)
        self.slider.setValue((self.warp_fun(value) - self.warp_fun(self.min_val)) / diff)
        if self.dtype == int:
            self.textBox.setValue('{:d}'.format(value))
        else:
            self.textBox.setValue('{:.3f}'.format(value))


def add_slider(variable_scope, widget, variable_name, min_val, max_val, dtype, callback=None, logspace=False):
    Label(widget, variable_name)
    widget = Widget(widget)
    widget.setLayout(BoxLayout(Orientation.Horizontal,
                               Alignment.Middle, 0, 20))

    slider = Slider(widget)
    diff = max_val - min_val
    logdiff = np.log(max_val) - np.log(min_val)
    if logspace:
        slider.setValue((np.log(variable_scope.__dict__[variable_name]) - np.log(min_val)) / logdiff)
    else:
        slider.setValue((variable_scope.__dict__[variable_name] - min_val) / diff)
    slider.setFixedWidth(120)

    textBox = TextBox(widget)
    textBox.setFixedSize((60, 25))
    textBox.setEditable(True)

    if dtype == int:
        textBox.setValue('{:d}'.format(variable_scope.__dict__[variable_name]))
    else:
        textBox.setValue('{:.3f}'.format(variable_scope.__dict__[variable_name]))

    textBox.setFontSize(20)
    textBox.setAlignment(TextBox.Alignment.Right)

    # TODO: Somehow the textbox callback is broken?
    # def cb_text(value):
    #     print('type(value): {}'.format(type(value)))
    #     print('value: {}'.format(value))
    # textBox.setCallback(cb_text)

    def cb(value):
        if logspace:
            if dtype == int:
                textBox.setValue('{:d}'.format(int(np.exp(logdiff * value + np.log(min_val)))))
            else:
                textBox.setValue('{:.3f}'.format(np.exp(logdiff * value + np.log(min_val))))
        else:
            if dtype == int:
                textBox.setValue('{:d}'.format(int(diff * value + min_val)))
            else:
                textBox.setValue('{:.3f}'.format(diff * value + min_val))

    slider.setCallback(cb)

    def cb(value):
        if logspace:
            variable_scope.__dict__[variable_name] = dtype(np.exp(logdiff * value + np.log(min_val)))
        else:
            variable_scope.__dict__[variable_name] = dtype(diff * value + min_val)
        if callback:
            callback()
    slider.setFinalCallback(cb)


def add_checkbox(variable_scope, widget, variable_name, initial_value, callback=None, label=None):
    def cb(value):
        variable_scope.__dict__[variable_name] = value
        if callback:
            callback()
    variable_scope.__dict__[variable_name] = initial_value
    checkbox = CheckBox(widget, label if label else variable_name)
    checkbox.setChecked(initial_value)
    checkbox.setCallback(cb)
    return checkbox