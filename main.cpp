// main
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

// OpenEngine stuff
#include <Meta/Config.h>

#include <Logging/Logger.h>
#include <Logging/StreamLogger.h>

#include <Scene/SceneNode.h>

#include <Core/Engine.h>

#include <Display/Viewport.h>
#include <Display/SDLEnvironment.h>
#include <Display/AntTweakBar.h>

#include <Renderers/IRenderingView.h>
#include <Renderers/OpenGL/MRIRenderingView.h>
#include <Renderers/TextureLoader.h>

#include <Math/Vector.h>

#include <Utils/SimpleSetup.h>
#include <Utils/BetterMoveHandler.h>
#include <Utils/CairoTextTool.h>
#include <Utils/InspectionBar.h>

#include "Science/MRIModule.h"

#include <Resources/CairoResource.h>
#include <Resources/DirectoryManager.h>
#include <Resources/ResourceManager.h>


using namespace OpenEngine::Core;
using namespace OpenEngine::Display;
using namespace OpenEngine::Logging;
using namespace OpenEngine::Math;
using namespace OpenEngine::Renderers;
using namespace OpenEngine::Renderers::OpenGL;
using namespace OpenEngine::Utils;
using namespace OpenEngine::Science;
using namespace OpenEngine::Scene;

class MRIHandler : public IListener<KeyboardEventArg> {
private:
    MRINode* node;
public:
    MRIHandler(MRINode* n) { node = n; }

    void Handle(KeyboardEventArg arg){
        if (arg.type == EVENT_PRESS && arg.sym == KEY_f)
            node->Flip(90);
    }
};


// Helpers
static TransformationNode* CreateTextureBillboard(ITextureResourcePtr texture,
                                                  float scale) {
    unsigned int textureHosisontalSize = texture->GetWidth();
    unsigned int textureVerticalSize = texture->GetHeight();

    logger.info << "w x h = " << texture->GetWidth()
                << " x " << texture->GetHeight() << logger.end;
    float fullxtexcoord = 1;
    float fullytexcoord = 1;
  
    FaceSet* faces = new FaceSet();

    float horisontalhalfsize = textureHosisontalSize * 0.5;
    Vector<3,float> lowerleft = Vector<3,float>(horisontalhalfsize,0,0);
    Vector<3,float> lowerright = Vector<3,float>(-horisontalhalfsize,0,0);
    Vector<3,float> upperleft = Vector<3,float>(horisontalhalfsize,textureVerticalSize,0);
    Vector<3,float> upperright = Vector<3,float>(-horisontalhalfsize,textureVerticalSize,0);

    FacePtr leftside = FacePtr(new Face(lowerleft,lowerright,upperleft));

    /*
      leftside->texc[1] = Vector<2,float>(1,0);
      leftside->texc[0] = Vector<2,float>(0,0);
      leftside->texc[2] = Vector<2,float>(0,1);
    */
    leftside->texc[1] = Vector<2,float>(0,fullytexcoord);
    leftside->texc[0] = Vector<2,float>(fullxtexcoord,fullytexcoord);
    leftside->texc[2] = Vector<2,float>(fullxtexcoord,0);
    leftside->norm[0] = leftside->norm[1] = leftside->norm[2] = Vector<3,float>(0,0,1);
    leftside->CalcHardNorm();
    leftside->Scale(scale);
    faces->Add(leftside);

    FacePtr rightside = FacePtr(new Face(lowerright,upperright,upperleft));
    /*
      rightside->texc[2] = Vector<2,float>(0,1);
      rightside->texc[1] = Vector<2,float>(1,1);
      rightside->texc[0] = Vector<2,float>(1,0);
    */
    rightside->texc[2] = Vector<2,float>(fullxtexcoord,0);
    rightside->texc[1] = Vector<2,float>(0,0);
    rightside->texc[0] = Vector<2,float>(0,fullytexcoord);
    rightside->norm[0] = rightside->norm[1] = rightside->norm[2] = Vector<3,float>(0,0,1);
    rightside->CalcHardNorm();
    rightside->Scale(scale);
    faces->Add(rightside);

    MaterialPtr m = leftside->mat = rightside->mat = MaterialPtr(new Material());
    m->AddTexture(texture);

    GeometryNode* node = new GeometryNode();
    node->SetFaceSet(faces);
    TransformationNode* tnode = new TransformationNode();
    tnode->AddNode(node);
    return tnode;
}

struct WallItem {
    ITextureResourcePtr texture;
    string title;
    Vector<2,unsigned int> scale;

    WallItem() {}

    WallItem(ITextureResourcePtr t, 
             string s)
        : texture(t)
        , title(s)
        , scale(Vector<2,unsigned int>(1,1)) {
        
    }
};

struct Wall {
    WallItem tex[12];
    TextureLoader& loader;

    Wall(TextureLoader& l) : loader(l) {
        
    }

    WallItem& operator()(int x, int y) {
        return tex[x*3+y];
    }
    ISceneNode* MakeScene() {
        SceneNode *sn = new SceneNode();
        CairoTextTool textTool;
        
        for (int x=0;x<4;x++) {
            for (int y=0;y<3;y++) {
                WallItem itm = (*this)(x,y);
                ITextureResourcePtr t = itm.texture;
                if (t) {
                    Vector<2,unsigned int> scale = itm.scale;
                    loader.Load(t,TextureLoader::RELOAD_QUEUED);
                    TransformationNode* node = new TransformationNode();
                    TransformationNode* bnode = CreateTextureBillboard(t,0.05);
                    bnode->SetScale(Vector<3,float>( 1.0 * scale[0],
                                                   -1.0 * scale[1],
                                                    1.0));
                    node->Move(x*35-25,y*35-25,0);
                    node->AddNode(bnode);
                    
                    CairoResourcePtr textRes = CairoResource::Create(128,32);
                    textRes->Load();

                    ostringstream out;
                    out << "(" << x << "," << y << ") " << itm.title;

                    textTool.DrawText(out.str(), textRes);

                    loader.Load(textRes);
                    TransformationNode* textNode = CreateTextureBillboard(textRes,0.15);

                    textNode->Move(0,-28,-0.01);


                    node->AddNode(textNode);
                    //sn->AddNode(textNode);
                    sn->AddNode(node);
                }
            }
        }

        return sn;
    }
};



/**
 * Main.
 */
int main(int argc, char** argv) {
    IEnvironment* env = new SDLEnvironment(1024,768,32);

    //IRenderingView* rv = new MRIRenderingView();
    IRenderingView* rv = NULL;

    // Create simple setup
    SimpleSetup* setup = new SimpleSetup("SpinOff", env, rv);
    
    DirectoryManager::AppendPath("./projects/spinOff/report/pics/");

    setup->GetRenderer().SetBackgroundColor(Vector<4,float>(0.1));


    BetterMoveHandler* move_h = new BetterMoveHandler(*(setup->GetCamera()), 
                                                      setup->GetMouse(),
                                                      true);

    setup->GetEngine().InitializeEvent().Attach(*move_h);
    setup->GetEngine().ProcessEvent().Attach(*move_h);
    setup->GetEngine().DeinitializeEvent().Attach(*move_h);

    // setup->GetCamera()->SetPosition(Vector<3, float>(20, 20, 0));

    // Ant tweak bar
    AntTweakBar *atb = new AntTweakBar();
    atb->AttachTo(setup->GetRenderer());

    atb->KeyEvent().Attach(*move_h);   
    atb->MouseButtonEvent().Attach(*move_h);
    atb->MouseMovedEvent().Attach(*move_h);


    
    setup->GetKeyboard().KeyEvent().Attach(*atb);
    setup->GetMouse().MouseMovedEvent().Attach(*atb);
    setup->GetMouse().MouseButtonEvent().Attach(*atb);


    // Graphics
    ITextureResourcePtr girl = ResourceManager<ITextureResource>::Create("frontpage.jpg");
    girl->Load();

    // Mri Module
    MRIModule* mri = new MRIModule(girl);
    setup->GetEngine().ProcessEvent().Attach(*mri);
    setup->GetEngine().InitializeEvent().Attach(*mri);
    setup->GetEngine().DeinitializeEvent().Attach(*mri);
    setup->GetKeyboard().KeyEvent().Attach(*mri);
    setup->GetRenderer().ProcessEvent().Attach(*mri);


    InspectionBar* mriBar = new InspectionBar("mri", Inspect(mri));
    mriBar->SetIconify(false);
    mriBar->SetPosition(Vector<2,float>(800,100));
    atb->AddBar(mriBar);

    // Wall
    Wall wall(setup->GetTextureLoader());

    wall(0,0) = WallItem(girl, "input");
    wall(1,0) = WallItem(mri->GetOutputTexture(), "output");
    wall(2,0) = WallItem(mri->GetInverseTexture(), "inverse");

    wall(0,1) = WallItem(mri->GetTestTexture(), "test output");
    wall(1,1) = WallItem(mri->GetDescaledTexture(), "descaled");
    wall(1,1).scale = Vector<2,unsigned int>(10,10);
    wall(2,1) = WallItem(mri->GetSignalTexture(), "signal");

    wall(1,2) = WallItem(mri->GetSignalOutputTexture(), "signal output");

    wall(0,2) = WallItem(mri->GetSignalOutput2Texture(), "signal output 2");
    wall(2,2) = WallItem(mri->GetPlotTexture(), "Plot");

    ISceneNode *wallNode = wall.MakeScene();
    setup->SetScene(*wallNode);

    float h = -25/2;
    setup->GetCamera()->SetPosition(Vector<3,float>(0.0,h,130));
    setup->GetCamera()->LookAt(Vector<3,float>(0.0,h,0.0));



    // Create Scene
    if (0) {
        MRINode* mrinode = new MRINode();
        setup->GetEngine().ProcessEvent().Attach(*mrinode);
        setup->SetScene(*mrinode);
        
        MRIHandler handle = MRIHandler(mrinode);
        setup->GetKeyboard().KeyEvent().Attach(handle);

        setup->GetCamera()->LookAt(0, 0, 0);
    }
    // Start the engine.
    setup->GetEngine().Start();    
    
    // Return when the engine stops.
    return EXIT_SUCCESS;
}


