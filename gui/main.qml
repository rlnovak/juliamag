// JuliaMag GUI — a dedicated Qt6/QML window (not a browser), initially laid out
// like mumax3: a control column on the left (mesh, material, field, solver,
// run/pause), and on the right a live Makie viewport showing the magnetization
// and the ⟨m⟩(t) curves. The layout is intentionally simple; restyle freely.

import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import org.julialang

ApplicationWindow {
    id: win
    visible: true
    width: 1100
    height: 720
    title: "JuliaMag"

    RowLayout {
        anchors.fill: parent
        anchors.margins: 8
        spacing: 8

        // ---- Control column ------------------------------------------------
        ColumnLayout {
            Layout.preferredWidth: 300
            Layout.fillHeight: true
            spacing: 6

            GroupBox {
                title: "Mesh"
                Layout.fillWidth: true
                GridLayout {
                    columns: 2
                    anchors.fill: parent
                    Label { text: "Nx" }  SpinBox { id: nx; from: 1; to: 4096; value: 160 }
                    Label { text: "Ny" }  SpinBox { id: ny; from: 1; to: 4096; value: 40 }
                    Label { text: "Nz" }  SpinBox { id: nz; from: 1; to: 512;  value: 1 }
                    Label { text: "cell (nm)" }
                    TextField { id: cell; text: "3.125"; Layout.fillWidth: true }
                }
            }

            GroupBox {
                title: "Material"
                Layout.fillWidth: true
                ColumnLayout {
                    anchors.fill: parent
                    ComboBox {
                        id: matbox
                        Layout.fillWidth: true
                        model: materials            // filled from Julia
                    }
                    GridLayout {
                        columns: 2
                        Label { text: "α (damping)" }
                        TextField { id: alpha; text: "0.02"; Layout.fillWidth: true }
                    }
                }
            }

            GroupBox {
                title: "Applied field (mT)"
                Layout.fillWidth: true
                GridLayout {
                    columns: 2
                    anchors.fill: parent
                    Label { text: "Bx" }  TextField { id: bx; text: "-24.6"; Layout.fillWidth: true }
                    Label { text: "By" }  TextField { id: by; text: "4.3";   Layout.fillWidth: true }
                    Label { text: "Bz" }  TextField { id: bz; text: "0.0";   Layout.fillWidth: true }
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
                title: "Run"
                Layout.fillWidth: true
                GridLayout {
                    columns: 2
                    anchors.fill: parent
                    Label { text: "duration (ns)" }
                    TextField { id: dur; text: "1.0"; Layout.fillWidth: true }
                    Label { text: "save Δt (ps)" }
                    TextField { id: savedt; text: "5.0"; Layout.fillWidth: true }
                }
            }

            RowLayout {
                Layout.fillWidth: true
                Button {
                    text: "Relax"
                    Layout.fillWidth: true
                    onClicked: Julia.gui_relax(nx.value, ny.value, nz.value,
                        cell.text, matbox.currentText, alpha.text, statebox.currentText)
                }
                Button {
                    text: "Run"
                    Layout.fillWidth: true
                    onClicked: Julia.gui_run(dur.text, savedt.text, bx.text, by.text, bz.text)
                }
            }
            Button {
                text: "Stop"
                Layout.fillWidth: true
                onClicked: Julia.gui_stop()
            }

            Label {
                id: statuslabel
                Layout.fillWidth: true
                wrapMode: Text.WordWrap
                text: status                        // observable from Julia
            }

            Item { Layout.fillHeight: true }        // spacer
        }

        // ---- Visualization column -----------------------------------------
        ColumnLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true

            MakieViewport {
                id: viewport
                Layout.fillWidth: true
                Layout.fillHeight: true
                scene: plot                         // Makie figure from Julia
            }
        }
    }
}
