using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using RedRunner.Utilities;


namespace RedRunner.UI
{
    public class UITimer : Text
    {
        [SerializeField]
        protected string m_TimerTextFormat = "Time left: {0:0}:{1:0}";
        [SerializeField]
        protected string m_TimerTextFormat2 = "Time left: {0:0}:0{1:0}";

        protected override void Awake()
        {
            GameManager.OnTimerChanged += GameManager_OnTimerChanged;
            GameManager.OnReset += GameManager_OnReset;
            base.Awake();
        }
        void GameManager_OnTimerChanged(float new_time)
        {
            float minutes = Mathf.Floor(new_time / 60);
            float seconds = Mathf.Floor(new_time - (minutes * 60));
            if (seconds < 10)
                text = string.Format(m_TimerTextFormat2, minutes, seconds);
            else text = string.Format(m_TimerTextFormat, minutes, seconds);
        }
        void GameManager_OnReset()
        {
            float minutes = Mathf.Floor(GameManager.Singleton.max_game_time / 60);
            float seconds = Mathf.Floor(GameManager.Singleton.max_game_time - (minutes * 60));
            if (seconds < 10)
                text = string.Format(m_TimerTextFormat2, minutes, seconds);
            else text = string.Format(m_TimerTextFormat, minutes, seconds);
        }
    }
}

