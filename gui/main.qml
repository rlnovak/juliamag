// JuliaMag GUI: a control column on the left; on the right the curves (⟨m⟩ and the energy),
// the state of the run, and one layer of m. A timer polls Julia for new data; the simulation
// itself runs on a Julia worker thread, so the window never waits for it.

import QtQuick
import QtQuick.Controls
import QtQuick.Dialogs
import QtQuick.Layouts
import jlqml

ApplicationWindow {
    id: win
    visible: true
    width: 1280
    height: 860
    title: "JuliaMag"

    Timer {
        interval: pollms; running: true; repeat: true
        onTriggered: {
            var r = Julia.gui_poll()        // which viewports changed: 1 curves, 2 the map
            if (r & 1) curveview.update()
            if (r & 2) fieldview.update()
        }
    }

    // JULIAMAG_GUI_SMOKE (app.jl): save a picture of the window when Julia asks, then quit
    property bool shot: props.shot
    onShotChanged: if (shot && !rootrow.grabToImage(function (r) { r.saveToFile(smoke); Qt.quit() })) Qt.quit()

    FileDialog {
        id: opendlg
        title: "Open a Julia script that defines sim"
        nameFilters: ["Julia scripts (*.jl)"]
        onAccepted: Julia.gui_loadscript(selectedFile.toString())
    }

    // Drop a .jl script anywhere on the window to load its `sim`.
    DropArea {
        anchors.fill: parent
        onDropped: (d) => {
            if (d.hasUrls && d.urls.length > 0)
                Julia.gui_loadscript(d.urls[0].toString())
        }
    }

    RowLayout {
        id: rootrow
        anchors.fill: parent
        anchors.margins: 8
        spacing: 8

        // ================= Controls =================
        ScrollView {
            Layout.preferredWidth: 300
            Layout.fillHeight: true
            contentWidth: availableWidth

            ColumnLayout {
                width: parent.width
                spacing: 6

                GroupBox {
                    title: "Script"
                    Layout.fillWidth: true
                    ColumnLayout {
                        anchors.fill: parent
                        Button {
                            text: "Open script…"; Layout.fillWidth: true
                            enabled: !props.busy
                            onClicked: opendlg.open()
                        }
                        Label {
                            Layout.fillWidth: true; wrapMode: Text.WordWrap; color: "#666"
                            text: "A .jl file that defines sim::Simulation (or drop it on the window)."
                        }
                    }
                }

                GroupBox {
                    title: "Or build one"
                    Layout.fillWidth: true
                    GridLayout {
                        columns: 2
                        anchors.fill: parent
                        Label { text: "cells x, y, z" }
                        RowLayout {
                            SpinBox { id: nx; from: 1; to: 4096; value: 64; editable: true; Layout.preferredWidth: 70 }
                            SpinBox { id: ny; from: 1; to: 4096; value: 64; editable: true; Layout.preferredWidth: 70 }
                            SpinBox { id: nz; from: 1; to: 256; value: 1; editable: true; Layout.preferredWidth: 60 }
                        }
                        Label { text: "cell (nm)" }
                        TextField { id: cell; text: "4"; Layout.fillWidth: true }
                        Label { text: "periodic x" }
                        CheckBox { id: pbcx }
                        Label { text: "shape" }
                        ComboBox { id: geombox; Layout.fillWidth: true
                                   model: ["Full mesh", "Cylinder", "Rectangle", "Ellipse 2:1"] }
                        Label { text: "size (nm)" }
                        TextField { id: geomsize; text: "256"; Layout.fillWidth: true
                                    enabled: geombox.currentIndex > 0 }
                        Label { text: "material" }
                        ComboBox { id: matbox; Layout.fillWidth: true; model: materials }
                        Label { text: "α" }
                        TextField { id: alpha; text: "0.02"; Layout.fillWidth: true }
                        Label { text: "initial m" }
                        ComboBox { id: statebox; Layout.fillWidth: true
                                   model: ["Uniform +x", "Vortex", "Néel skyrmion", "Bloch skyrmion", "Random"] }
                    }
                }
                Button {
                    text: "Build"; Layout.fillWidth: true
                    enabled: !props.busy
                    onClicked: Julia.gui_build(nx.value, ny.value, nz.value, cell.text, matbox.currentText,
                                               alpha.text, statebox.currentText, geombox.currentText,
                                               geomsize.text, pbcx.checked)
                }

                GroupBox {
                    title: "Applied field (mT)"
                    Layout.fillWidth: true
                    RowLayout {
                        anchors.fill: parent
                        TextField { id: bx; text: "0"; Layout.fillWidth: true }
                        TextField { id: by; text: "0"; Layout.fillWidth: true }
                        TextField { id: bz; text: "0"; Layout.fillWidth: true }
                        Button { text: "Set"; onClicked: Julia.gui_field(bx.text, by.text, bz.text) }
                    }
                }

                GroupBox {
                    title: "Run"
                    Layout.fillWidth: true
                    ColumnLayout {
                        anchors.fill: parent
                        RowLayout {
                            Label { text: "duration (ns)" }
                            TextField { id: dur; text: "1"; Layout.fillWidth: true
                                        placeholderText: "empty: until Stop" }
                        }
                        RowLayout {
                            Layout.fillWidth: true
                            Button { text: "Relax"; Layout.fillWidth: true; enabled: !props.busy
                                     onClicked: Julia.gui_relax() }
                            Button { text: "Run"; Layout.fillWidth: true; enabled: !props.busy
                                     onClicked: Julia.gui_run(dur.text) }
                        }
                        RowLayout {
                            Layout.fillWidth: true
                            Button { text: props.paused ? "Resume" : "Pause"; Layout.fillWidth: true
                                     enabled: props.running; onClicked: Julia.gui_pause() }
                            Button { text: "Stop"; Layout.fillWidth: true; enabled: props.running
                                     onClicked: Julia.gui_stop() }
                        }
                    }
                }

                Label {
                    Layout.fillWidth: true
                    wrapMode: Text.WordWrap
                    text: props.status
                }
                Item { Layout.fillHeight: true }
            }
        }

        // ================= Display =================
        ColumnLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 6

            RowLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                Layout.preferredHeight: 1
                MakieViewport {
                    id: curveview
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    Layout.minimumWidth: 300
                    scene: plot_curve
                }
                ColumnLayout {
                    Layout.preferredWidth: 380
                    Layout.maximumWidth: 380
                    Layout.fillHeight: true
                    Label {
                        Layout.fillWidth: true
                        font.family: "Consolas"
                        font.pixelSize: 13
                        wrapMode: Text.Wrap
                        text: props.info
                    }
                    CheckBox { id: energyon; text: "energy curve"; checked: true
                               onToggled: { Julia.gui_view(viewbox.currentText, layerbox.value, checked); fieldview.update() } }
                    Label { color: "#888"; text: props.fps }
                    Item { Layout.fillHeight: true }
                }
            }

            ColumnLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                Layout.preferredHeight: 1.3
                MakieViewport {
                    id: fieldview
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    Layout.minimumHeight: 200
                    scene: plot_field
                }
                RowLayout {
                    Layout.fillWidth: true
                    Label { text: "show:" }
                    ComboBox {
                        id: viewbox
                        model: ["wheel", "mz", "mx", "my"]
                        onActivated: { Julia.gui_view(currentText, layerbox.value, energyon.checked); fieldview.update() }
                    }
                    Label { text: "layer:" }
                    SpinBox {
                        id: layerbox
                        from: 1; to: props.nz; value: 1; editable: true
                        onValueModified: { Julia.gui_view(viewbox.currentText, value, energyon.checked); fieldview.update() }
                    }
                    Item { Layout.fillWidth: true }
                }
            }
        }
    }
}
