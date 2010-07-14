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

namespace OpenEngine {
namespace Science {

/**
 * Short description.
 *
 * @class MRIModule MRIModule.h ons/SpinOff/Science/MRIModule.h
 */

using namespace Core;
using namespace Resources;

class MRIModule : public IModule
                , public IListener<Devices::KeyboardEventArg> {
private:
    ITextureResourcePtr img;
    EmptyTextureResourcePtr outputTexture;
    EmptyTextureResourcePtr inverseTexture;

    bool running;
    bool test;
    float b0;
public:
    MRIModule(ITextureResourcePtr img);
    void Handle(ProcessEventArg arg);
    void Handle(InitializeEventArg arg);
    void Handle(DeinitializeEventArg arg);
    void Handle(Devices::KeyboardEventArg arg);
    
    EmptyTextureResourcePtr GetOutputTexture() { return outputTexture; }
    EmptyTextureResourcePtr GetInverseTexture() { return inverseTexture; }

    Utils::Inspection::ValueList Inspection();

    bool GetTest() { return test; }
    void SetTest(bool b) {test = b ;}

    bool GetRunning() { return running; }
    void SetRunning(bool running) { this->running = running; }

    float GetB0() { return b0; }
    void SetB0(float b0) { this->b0 = b0; }

};

} // NS Science

namespace Utils {
namespace Inspection {
    
    ValueList Inspect(Science::MRIModule *);
    
}
}


} // NS OpenEngine

#endif // _OE_MRI_MODULE_H_
