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

        public void ChangeHeart()
        {
            GetComponent<Animator>().SetTrigger("Reverse");
        }
    }
}

