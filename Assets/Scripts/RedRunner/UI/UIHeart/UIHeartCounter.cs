using System.Collections;
using System.Collections.Generic;
using UnityEngine;
namespace RedRunner.UI
{
    public class UIHeartCounter : MonoBehaviour
    {
        public UIHeartsImage heart1;
        public UIHeartsImage heart2;
        public UIHeartsImage heart3;
        private int lives = 3;
        protected void Start()
        {
            RedRunner.Characters.RedCharacter.OnHeartLoss += RedCharacter_OnHeartLoss;
            RedRunner.Characters.RedCharacter.OnHeartReset += RedCharacter_OnHeartReset;
            lives = 3;
        }

        void RedCharacter_OnHeartLoss()
        {
            switch (lives)
            {
                case 3:
                    heart3.ChangeHeart();
                    break;
                case 2:
                    heart2.ChangeHeart();
                    break;
                case 1:
                    heart1.ChangeHeart();
                    break;
                default:
                    break;
            }
            lives--;
        }

        void RedCharacter_OnHeartReset()
        {
            lives = 3;
            heart3.ChangeHeart();
            heart2.ChangeHeart();
            heart1.ChangeHeart();
        }
    }
}