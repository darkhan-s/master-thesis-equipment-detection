using System;
using System.Drawing;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using wf = System.Windows.Forms;
using System.Text.RegularExpressions;

using ComApi = Autodesk.Navisworks.Api.Interop.ComApi;
using Autodesk.Navisworks.Api;
using Autodesk.Navisworks.Api.DocumentParts;
using Autodesk.Navisworks.Api.Plugins;
using Autodesk.Navisworks.Api.Automation;
using ComApiBridge = Autodesk.Navisworks.Api.ComApi;

namespace ImageDataExport
{
    [PluginAttribute("BasicPlugIn.ImageDataExport",                   //Plugin name
                 "SAID",                                       //4 character Developer ID or GUID
                 ToolTip = "Export images of the selected models one by one for each angle and record the corresponding locations",//The tooltip for the item in the ribbon
                 DisplayName = "Export images and properties")]          //Display name for the Plugin in the Ribbon


    public class ImageDataExportPlugin : AddInPlugin
    {
        public override int Execute(params string[] parameters)
        {
            FindByName();
            
            //wf.MessageBox.Show("Done", "API message");
            return 0;
        }


        public void FindByName()
        {
            // display message
            StringBuilder message = new StringBuilder();
            // current document
            Document doc = Application.ActiveDocument;

            // create search object
            Search search = new Search();
            // selection to search
            search.Selection.SelectAll();

            string searchValue = "/3410-REA-001/N11";
            string searchValue2 = "EXTRUSION 2 of SUBEQUIPMENT / 3410 - REA - 001 / NOZZLE";
       
            SearchCondition condition = SearchCondition.HasPropertyByDisplayName("Item",
                 "Name").EqualValue(new VariantData(searchValue));

            // SearchCondition to applied during search
            search.SearchConditions.Add(condition);
            // collect model item (if found)
            //var items = search.FindAll(doc, false);
            //ModelItem item = items[0];
            ModelItem item = search.FindFirst(doc, false);

            // item found
            if (item != null)
            {
                // make selection
                doc.CurrentSelection.Add(item);
                FocusOnObject();
            
                int resolution = 20;
                for (int i = 0; i < resolution; i++)
                {
                    SpinModel(resolution);
                    SaveImage($"{searchValue}_{i}");
                }
            }
            else
            {
                // display message
                wf.MessageBox.Show("Item not found");

            }
        }
        public void HideUnselected() 
        {
            
        }
        public void FocusOnObject()
        {

            // https://adndevblog.typepad.com/aec/2012/05/how-to-zoom-in-current-view-on-current-selection-with-net-api.html 
            //var view = doc.ActiveView;
            //view.FocusOnCurrentSelection();

            ComApi.InwOpState10 comState = ComApiBridge.ComApiBridge.State;
            //Create a collection
            ModelItemCollection modelItemCollectionIn =
                new ModelItemCollection(
                    Autodesk.Navisworks.Api.Application.ActiveDocument
                    .CurrentSelection.SelectedItems);
            ComApi.InwOpSelection comSelectionOut =
                ComApiBridge.ComApiBridge.ToInwOpSelection(modelItemCollectionIn);
            // zoom in to the specified selection
            comState.ZoomInCurViewOnSel(comSelectionOut);
            // zoom in to the current selection
            //comState.ZoomInCurViewOnCurSel();
        }
        public void SpinModel(int resolution)
        {
            // https://adndevblog.typepad.com/aec/2012/06/navisworks-net-api-2013-new-feature-viewpoint-3.html
            Document oDoc = Autodesk.Navisworks.Api.Application.ActiveDocument;
            
            //  Make a copy of current viewpoint   
            Viewpoint oCurrVCopy = oDoc.CurrentViewpoint.CreateCopy();
            //  set the axis we will rotate around （¨Ｚú：o０°,０°,１±）?
            UnitVector3D odeltaA = new UnitVector3D(0, 0, 1);

            // Create delta of Quaternion: axis is Z,
            // angle is 45 degree
            Rotation3D delta = new Rotation3D(odeltaA, 3.14 / resolution);

            // multiply the current Quaternion with the delta ,
            // get the new Quaternion
            oCurrVCopy.Rotation = MultiplyRotation3D(delta, oCurrVCopy.Rotation);

            // Update current viewpoint
            oDoc.CurrentViewpoint.CopyFrom(oCurrVCopy);

        }
        private Rotation3D MultiplyRotation3D(Rotation3D r2,Rotation3D r1)

        {
            Rotation3D oRot =
                new Rotation3D(r2.D * r1.A + r2.A * r1.D +
                                    r2.B * r1.C - r2.C * r1.B, r2.D * r1.B + r2.B * r1.D + r2.C * r1.A - r2.A * r1.C, r2.D * r1.C + r2.C * r1.D +

                                    r2.A * r1.B - r2.B * r1.A, r2.D * r1.D - r2.A * r1.A - r2.B * r1.B - r2.C * r1.C);
            oRot.Normalize();
            return oRot;
        }
        /// <summary>
        /// Saving the image and its data with the object of interest
        /// </summary>
        public void SaveImage(string name)
        {
            Document doc = Application.ActiveDocument;
            var view = doc.ActiveView;
            var bmp = view.GenerateThumbnail(view.Width, view.Height);
            // only space, capital A-Z, lowercase a-z, and digits 0-9 are allowed in the string
            string cleanName = Regex.Replace(name, "[^A-Za-z0-9 ]", "");

            bmp.Save($"{cleanName}.png", System.Drawing.Imaging.ImageFormat.Png);

            SaveCoordinates(name);


        }

        /// <summary>
        /// Method to extract viewpoint data (to later predict the real object viewpoint)
        /// </summary>
        /// <returns></returns>
        public Point3D SaveCoordinates(string name)
        {
            //more here
            //https://twentytwo.space/2020/12/06/navisworks-api-viewpoint-part-1/
            Document doc = Application.ActiveDocument;
            Viewpoint vpoint = doc.CurrentViewpoint.CreateCopy();

            Point3D currentCameraPos = vpoint.Position;
            return currentCameraPos;
        }

        /// <summary>
        /// Method to switch to the predicted viewpoint
        /// </summary>
        /// <returns></returns>
        public void MoveToViewpoint(Point3D target)
        {
            Document doc = Application.ActiveDocument;
            Viewpoint vpoint = doc.CurrentViewpoint.CreateCopy();

            vpoint.Position = target;
            doc.CurrentViewpoint.CopyFrom(vpoint);


        }
    }
}
