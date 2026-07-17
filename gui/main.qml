// JuliaMag GUI — a dedicated Qt6/QML window (not a browser). A control column on
// the left (mesh, cells, geometry, material, field, run controls, script drop),
// and on the right two Makie viewports: a time-series panel with a y-quantity
// dropdown, and a field panel (mz colour map + in-plane quiver) with quantity,
// colormap, and layer dropdowns and an on/off checkbox.

import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import jlqml

ApplicationWindow {
    id: win
    visible: true
    width: 1240
    height: 820
    title: "JuliaMag"

    // Drop a .jl script anywhere on the window to load its `sim`.
    DropArea {
        id: drop
        anchors.fill: parent
        onDropped: (d) => {
            if (d.hasUrls && d.urls.length > 0)
                Julia.gui_loadscript(d.urls[0].toString())
        }
    }

    RowLayout {
        anchors.fill: parent
        anchors.margins: 8
        spacing: 8

        // ================= Control column =================
        ScrollView {
            Layout.preferredWidth: 320
            Layout.fillHeight: true
            contentWidth: availableWidth

            ColumnLayout {
                width: parent.width
                spacing: 6

                GroupBox {
                    title: "Mesh"
                    Layout.fillWidth: true
                    GridLayout {
                        columns: 2
                        anchors.fill: parent
                        Label { text: "Nx" }  SpinBox { id: nx; from: 1; to: 8192; value: 100 }
                        Label { text: "Ny" }  SpinBox { id: ny; from: 1; to: 8192; value: 100 }
                        Label { text: "Nz" }  SpinBox { id: nz; from: 1; to: 512;  value: 1 }
                        Label { text: "cell (nm)" }
                        TextField { id: cell; text: "5.0"; Layout.fillWidth: true }
                        Label { text: "periodic x" }
                        CheckBox { id: pbcx; checked: false }
                    }
                }

                GroupBox {
                    title: "Geometry"
                    Layout.fillWidth: true
                    GridLayout {
                        columns: 2
                        anchors.fill: parent
                        Label { text: "shape" }
                        ComboBox {
                            id: geombox
                            Layout.fillWidth: true
                            model: ["Full mesh", "Cylinder", "Rectangle", "Ellipse"]
                        }
                        Label { text: "size (nm)" }
                        TextField { id: geomsize; text: "300"; Layout.fillWidth: true
                                    enabled: geombox.currentIndex > 0 }
                    }
                }

                GroupBox {
                    title: "Material"
                    Layout.fillWidth: true
                    ColumnLayout {
                        anchors.fill: parent
                        ComboBox { id: matbox; Layout.fillWidth: true; model: materials }
                        GridLayout {
                            columns: 2
                            Label { text: "α (damping)" }
                            TextField { id: alpha; text: "0.02"; Layout.fillWidth: true }
                        }
                    }
                }

                GroupBox {
                    title: "Initial state"
                    Layout.fillWidth: true
                    ComboBox {
                        id: statebox
                        anchors.fill: parent
                        model: ["Uniform +x", "Vortex", "Néel skyrmion", "Bloch skyrmion", "Random"]
                    }
                }

                GroupBox {
                    title: "Applied field (mT)"
                    Layout.fillWidth: true
                    GridLayout {
                        columns: 2
                        anchors.fill: parent
                        Label { text: "Bx" }  TextField { id: bx; text: "0.0"; Layout.fillWidth: true }
                        Label { text: "By" }  TextField { id: by; text: "0.0"; Layout.fillWidth: true }
                        Label { text: "Bz" }  TextField { id: bz; text: "0.0"; Layout.fillWidth: true }
                    }
                }

                GroupBox {
                    title: "Run"
                    Layout.fillWidth: true
                    GridLayout {
                        columns: 2
                        anchors.fill: parent
                        Label { text: "duration (ns)" }
                        TextField { id: dur; text: "1.0"; Layout.fillWidth: true }
                        Label { text: "step Δt (ps)" }
                        TextField { id: savedt; text: "5.0"; Layout.fillWidth: true }
                    }
                }

                // Build / Relax
                RowLayout {
                    Layout.fillWidth: true
                    Button {
                        text: "Build"; Layout.fillWidth: true
                        onClicked: Julia.gui_build(nx.value, ny.value, nz.value, cell.text,
                            matbox.currentText, alpha.text, statebox.currentText,
                            geombox.currentText, geomsize.text, pbcx.checked)
                    }
                    Button {
                        text: "Relax"; Layout.fillWidth: true
                        onClicked: Julia.gui_relax()
                    }
                }
                // Run / Pause / Step / Stop
                RowLayout {
                    Layout.fillWidth: true
                    Button {
                        text: "Run"; Layout.fillWidth: true
                        onClicked: Julia.gui_run(dur.text, savedt.text, bx.text, by.text, bz.text)
                    }
                    Button {
                        text: "Pause"; Layout.fillWidth: true
                        onClicked: Julia.gui_pause()
                    }
                }
                RowLayout {
                    Layout.fillWidth: true
                    Button {
                        text: "Step"; Layout.fillWidth: true
                        onClicked: Julia.gui_step(savedt.text, bx.text, by.text, bz.text)
                    }
                    Button {
                        text: "Stop"; Layout.fillWidth: true
                        onClicked: Julia.gui_stop()
                    }
                }

                Label {
                    Layout.fillWidth: true
                    wrapMode: Text.WordWrap
                    color: "#555"
                    text: "Drop a .jl script defining `sim` onto the window to load it."
                }
                Label {
                    id: statuslabel
                    Layout.fillWidth: true
                    wrapMode: Text.WordWrap
                    text: status
                }

                Item { Layout.fillHeight: true }   // spacer
            }
        }

        // ================= Visualization column =================
        ColumnLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 8

            // ---- Panel 1: time series ----
            ColumnLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                MakieViewport {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    scene: plot_curve
                }
                RowLayout {
                    Layout.fillWidth: true
                    Label { text: "y-axis:" }
                    ComboBox {
                        id: ybox
                        Layout.fillWidth: true
                        model: ["mx", "my", "mz", "vortex x", "vortex y",
                                "skyrmion x", "skyrmion y", "topological charge"]
                        onActivated: Julia.gui_set_ycurve(currentText)
                    }
                }
            }

            // ---- Panel 2: field (colour map + quiver) ----
            ColumnLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                MakieViewport {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    scene: plot_field
                    visible: fieldon.checked
                }
                RowLayout {
                    Layout.fillWidth: true
                    CheckBox {
                        id: fieldon
                        text: "show"
                        checked: true
                        onToggled: Julia.gui_set_field(fqbox.currentText, cmapbox.currentText,
                                                       layerbox.value, checked)
                    }
                    Label { text: "map:" }
                    ComboBox {
                        id: fqbox
                        Layout.fillWidth: true
                        model: ["mz", "mx", "my"]
                        onActivated: Julia.gui_set_field(currentText, cmapbox.currentText,
                                                         layerbox.value, fieldon.checked)
                    }
                    Label { text: "cmap:" }
                    ComboBox {
                        id: cmapbox
                        Layout.fillWidth: true
                        model: ["balance", "viridis", "plasma", "coolwarm", "RdBu", "hot"]
                        onActivated: Julia.gui_set_field(fqbox.currentText, currentText,
                                                         layerbox.value, fieldon.checked)
                    }
                    Label { text: "layer:" }
                    SpinBox {
                        id: layerbox
                        from: 1; to: 512; value: 1
                        onValueModified: Julia.gui_set_field(fqbox.currentText, cmapbox.currentText,
                                                             value, fieldon.checked)
                    }
                }
            }
        }
    }
}
