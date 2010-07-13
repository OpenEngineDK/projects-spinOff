// MRI 4 Science!
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS)
//
// This program is free software; It is covered by the GNU General
// Public License version 2 or any later version.
// See the GNU General Public License for more details (see LICENSE).
//--------------------------------------------------------------------


#ifndef _OE_M_R_I_MODULE_H_
#define _OE_M_R_I_MODULE_H_

#include <Core/IModule.h>
#include <Devices/IKeyboard.h>
#include <Resources/ITexture2D.h>
#include <Resources/EmptyTextureResource.h>


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
public:
    MRIModule(ITextureResourcePtr img);
    void Handle(ProcessEventArg arg);
    void Handle(InitializeEventArg arg);
    void Handle(DeinitializeEventArg arg);
    void Handle(Devices::KeyboardEventArg arg);
    
    EmptyTextureResourcePtr GetOutputTexture() { return outputTexture; }
    EmptyTextureResourcePtr GetInverseTexture() { return inverseTexture; }

};

} // NS Science
} // NS OpenEngine

#endif // _OE_M_R_I_MODULE_H_
