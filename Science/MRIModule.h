// MRI 4 Science!
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS)
//
// This program is free software; It is covered by the GNU General
// Public License version 2 or any later version.
// See the GNU General Public License for more details (see LICENSE).
//--------------------------------------------------------------------


#ifndef _OE_MRI_MODULE_H_
#define _OE_MRI_MODULE_H_

#include <Core/IModule.h>
#include <Devices/IKeyboard.h>
#include <Resources/ITexture2D.h>
#include <Resources/EmptyTextureResource.h>
#include <Utils/IInspector.h>
#include <Renderers/IRenderer.h>
#include <Science/Plot.h>
#include <Science/PointGraphDataSet.h>
#include "MRI.hcu"

namespace OpenEngine {
namespace Science {

/**
 * Short description.
 *
 * @class MRIModule MRIModule.h ons/SpinOff/Science/MRIModule.h
 */

#define DESCALE_W 7
#define DESCALE_H DESCALE_W

#define SIGNAL_SIZE 500

using namespace Core;
using namespace Resources;

class MRIModule : public IModule
                , public IListener<Devices::KeyboardEventArg>
                , public IListener<Renderers::RenderingEventArg> {
private:
    ITextureResourcePtr img;
    EmptyTextureResourcePtr outputTexture;
    EmptyTextureResourcePtr inverseTexture;
    EmptyTextureResourcePtr testOutputTexture;
    EmptyTextureResourcePtr descaledOutputTexture;
    EmptyTextureResourcePtr signalTexture;
    EmptyTextureResourcePtr signalOutputTexture;
    EmptyTextureResourcePtr signalOutput2Texture;
    EmptyTextureResourcePtr plotTexture;
    
    Plot* plot;
    PointGraphDataSet* plotData1;
    PointGraphDataSet* plotData2;

    Vector<3,float> descaledVectors[DESCALE_W][DESCALE_H];


    

    bool running, fid, sequence, test;
    float relax;
    float b0, gx, gy, fov, phaseTime;
    float *lab_spins, *ref_spins;
    SpinProperty* props;
    unsigned int idx;


    
    float theDT, theTime;
    Vector<2,int> sigIdx;
    unsigned int sig2Idx;
    
    cuFloatComplex *signalData;

    void Descale(float *data, int w, int h);
    void FIDSequence();

public:
    MRIModule(ITextureResourcePtr img);
    void Handle(ProcessEventArg arg);
    void Handle(InitializeEventArg arg);
    void Handle(DeinitializeEventArg arg);
    void Handle(Devices::KeyboardEventArg arg);
    void Handle(Renderers::RenderingEventArg arg);
    
    EmptyTextureResourcePtr GetOutputTexture() { return outputTexture; }
    EmptyTextureResourcePtr GetInverseTexture() { return inverseTexture; }
    EmptyTextureResourcePtr GetTestTexture() { return testOutputTexture; }
    EmptyTextureResourcePtr GetDescaledTexture() { return descaledOutputTexture; }
    EmptyTextureResourcePtr GetSignalTexture() { return signalTexture; }
    EmptyTextureResourcePtr GetSignalOutputTexture() { return signalOutputTexture; }
    EmptyTextureResourcePtr GetSignalOutput2Texture() { return signalOutput2Texture; }
    EmptyTextureResourcePtr GetPlotTexture() { return plotTexture; }

    Utils::Inspection::ValueList Inspection();

    bool GetTest() { return test; }
    void SetTest(bool b) {test = b ;}

    bool GetRunning() { return running; }
    void SetRunning(bool running) { this->running = running; }

    bool GetSequence() { return sequence; }
    void SetSequence(bool sequence) { this->sequence = sequence; }

    float GetB0() { return b0; }
    void SetB0(float b0) { this->b0 = b0; }

    float GetGx() { return gx; }
    void SetGx(float gx) { this->gx = gx; }

    float GetGy() { return gy; }
    void SetGy(float gy) { this->gy = gy; }

    float GetFOV() { return fov; }
    void SetFOV(float fov) { this->fov = fov; }

    float GetPhaseTime() { return phaseTime; }
    void SetPhaseTime(float phaseTime) { this->phaseTime = phaseTime; }

    bool GetFID() { return fid; }
    void SetFID(bool enable) { fid = enable; }

    float GetDT() { return theDT; }
    void SetDT(float d) { theDT = d; }

    unsigned int GetIndex() { return idx; }
    void SetIndex(unsigned int index) { idx = index; }

    
};

} // NS Science

namespace Utils {
namespace Inspection {
    
    ValueList Inspect(Science::MRIModule *);
    
}
}


} // NS OpenEngine

#endif // _OE_MRI_MODULE_H_
