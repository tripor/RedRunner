using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace RedRunner.UI
{
    public class UIHeartsImage : Image
    {
        protected override void Awake()
        {
        }

        public void EmptyHeart()
        {
            GetComponent<Animator>().SetTrigger("Empty");
        }
        public void FillHeart()
        {
            GetComponent<Animator>().SetTrigger("Fill");
        }
        public void ResetHeart()
        {
            GetComponent<Animator>().SetTrigger("Reset");
        }
    }
}

